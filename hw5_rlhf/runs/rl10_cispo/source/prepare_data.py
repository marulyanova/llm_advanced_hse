import hashlib
import json
import random
from collections import Counter
from statistics import mean, median

from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from config import CONFIG as COMMON_CONFIG, project_path
from rewards import ToolReward
from toolace import format_calls, unfold_dialogue
from utils import file_hash, package_versions, write_json, write_jsonl


CONFIG = {
    "dataset_id": "Team-ACE/ToolACE",
    "dataset_revision": "main",
    "dataset_split": "train",
    "validation_fraction": 0.1,
    "max_prompt_tokens": 3072,
    "max_reference_tokens": 1024,
    "histogram_bins": 20,
    "top_tools": 20,
    "reward_examples_per_kind": 1,
}


def load_source(data_dir):
    revision = HfApi().dataset_info(
        CONFIG["dataset_id"], revision=CONFIG["dataset_revision"]
    ).sha
    dataset = load_dataset(
        CONFIG["dataset_id"], revision=revision, split=CONFIG["dataset_split"]
    )
    dataset.save_to_disk(str(data_dir / "raw"))
    return dataset, revision


def split_dialogues(dataset):
    unique = {}
    for index, row in enumerate(dataset):
        content = json.dumps(row, sort_keys=True, ensure_ascii=False)
        dialogue_id = hashlib.sha256(content.encode()).hexdigest()
        if dialogue_id not in unique:
            unique[dialogue_id] = index
    dialogue_ids = sorted(unique)
    random.Random(COMMON_CONFIG["seed"]).shuffle(dialogue_ids)
    count = max(1, int(len(dialogue_ids) * CONFIG["validation_fraction"]))
    validation = set(dialogue_ids[:count])
    return [
        {"dialogue_id": key, "source_index": unique[key],
         "split": "validation" if key in validation else "train"}
        for key in dialogue_ids
    ]


def make_row(example, source, tokenizer):
    prompt = tokenizer.apply_chat_template(
        example["messages"], tools=example["tools"], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    reference_tokens = 1 + len(tokenizer.encode(example["reference"], add_special_tokens=False))
    return {
        "id": f"{source['dialogue_id']}:{example['turn_index']}",
        "dialogue_id": source["dialogue_id"],
        "source_index": source["source_index"],
        "turn_index": example["turn_index"],
        "prompt": prompt,
        "tools_json": json.dumps(example["tools"], ensure_ascii=False),
        "expected_json": json.dumps(example["calls"], ensure_ascii=False),
        "reference": example["reference"],
        "prompt_tokens": prompt_tokens,
        "reference_tokens": reference_tokens,
        "tool_count": len(example["tools"]),
        "call_count": len(example["calls"]),
        "kind": "call" if example["calls"] else "no_call",
    }


def filter_reason(row, scorer):
    if row["prompt_tokens"] > CONFIG["max_prompt_tokens"]:
        return "long_prompt"
    if row["reference_tokens"] > CONFIG["max_reference_tokens"]:
        return "long_reference"
    score = scorer.score(
        row["reference"], json.loads(row["tools_json"]), json.loads(row["expected_json"])
    )
    if score["exact"] != 1.0:
        return "reference_not_roundtrippable"
    return None


def prepare_rows(dataset, sources, tokenizer, scorer):
    rows = {"train": [], "validation": []}
    candidates = {"train": [], "validation": []}
    rejected = []
    for source in sources:
        try:
            examples = unfold_dialogue(dataset[source["source_index"]])
        except (ValueError, SyntaxError, TypeError, KeyError) as error:
            rejected.append({**source, "reason": "dialogue_parse", "detail": str(error)})
            continue
        for example in examples:
            row = make_row(example, source, tokenizer)
            candidates[source["split"]].append(row)
            reason = filter_reason(row, scorer)
            if reason:
                rejected.append({"id": row["id"], "split": source["split"], "reason": reason})
            else:
                rows[source["split"]].append(row)
    return rows, candidates, rejected


def length_summary(values):
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    width = max(1, (max(values) + CONFIG["histogram_bins"]) // CONFIG["histogram_bins"])
    histogram = Counter(value // width for value in values)
    return {
        "count": len(values), "min": min(values), "mean": mean(values),
        "median": median(values), "p95": ordered[int(0.95 * (len(values) - 1))],
        "max": max(values),
        "histogram": [{"left": index * width, "right": (index + 1) * width,
                       "count": histogram[index]}
                      for index in range(max(histogram) + 1)],
    }


def describe(rows):
    tools = Counter()
    for row in rows:
        tools.update(tool["function"]["name"] for tool in json.loads(row["tools_json"]))
    return {
        "examples": len(rows),
        "dialogues": len({row["dialogue_id"] for row in rows}),
        "kinds": dict(Counter(row["kind"] for row in rows)),
        "calls_per_answer": dict(Counter(row["call_count"] for row in rows)),
        "tools_per_prompt": length_summary([row["tool_count"] for row in rows]),
        "prompt_tokens": length_summary([row["prompt_tokens"] for row in rows]),
        "reference_tokens": length_summary([row["reference_tokens"] for row in rows]),
        "unique_tool_names": len(tools),
        "top_tool_names": tools.most_common(CONFIG["top_tools"]),
    }


def save_reward_examples(rows, scorer, path):
    selected = []
    for kind in ("call", "no_call"):
        selected.extend([row for row in rows if row["kind"] == kind]
                        [:CONFIG["reward_examples_per_kind"]])
    examples = []
    for row in selected:
        tools, expected = json.loads(row["tools_json"]), json.loads(row["expected_json"])
        cases = [("reference", row["reference"], False),
                 ("empty", "", False), ("truncated", row["reference"], True)]
        if row["kind"] == "call":
            cases.append(("missing_call", "Please clarify your request.", False))
            cases.append(("duplicate_call", format_calls(expected + expected[:1]), False))
            incomplete = json.loads(row["expected_json"])
            if incomplete[0]["arguments"]:
                incomplete[0]["arguments"].pop(next(iter(incomplete[0]["arguments"])))
                cases.append(("missing_argument", format_calls(incomplete), False))
        else:
            extra = [{"name": tools[0]["function"]["name"], "arguments": {}}]
            cases.append(("extra_call", format_calls(extra), False))
        for case, text, truncated in cases:
            examples.append({
                "id": row["id"], "kind": row["kind"], "case": case,
                "expected": expected, "response": text, "truncated": truncated,
                "scores": scorer.score(text, tools, expected, truncated),
            })
    write_jsonl(path, examples)


def save_outputs(data_dir, dataset, revision, sources, rows, candidates, rejected):
    for split, examples in rows.items():
        if not examples:
            raise ValueError(f"Сплит {split} пуст после фильтрации; см. data/rejected.jsonl")
        write_jsonl(data_dir / f"{split}.jsonl", examples)
    write_json(data_dir / "eda.json", {
        "raw_dialogues": len(dataset), "unique_dialogues": len(sources),
        "duplicate_dialogues": len(dataset) - len(sources),
        "rejected": dict(Counter(row["reason"] for row in rejected)),
        "before_filtering": {key: describe(value) for key, value in candidates.items()},
        "after_filtering": {key: describe(value) for key, value in rows.items()},
    })
    write_json(data_dir / "manifest.json", {
        "config": CONFIG, "common_config": COMMON_CONFIG,
        "dataset_revision": revision, "dataset_fingerprint": dataset._fingerprint,
        "versions": package_versions(),
        "files": {f"{split}.jsonl": file_hash(data_dir / f"{split}.jsonl") for split in rows},
        "source_code": {name: file_hash(COMMON_CONFIG["root"] / name)
                        for name in ("prepare_data.py", "toolace.py", "rewards.py", "config.py")},
    })


def main():
    data_dir = project_path("data_dir")
    data_dir.mkdir(parents=True, exist_ok=True)
    if any((data_dir / name).exists() for name in ("train.jsonl", "raw")):
        raise FileExistsError("Данные уже есть. Для новой подготовки выберите другой data_dir.")
    dataset, revision = load_source(data_dir)
    sources = split_dialogues(dataset)
    write_jsonl(data_dir / "dialogue_splits.jsonl", sources)
    tokenizer = AutoTokenizer.from_pretrained(
        COMMON_CONFIG["model_id"], revision=COMMON_CONFIG["model_revision"]
    )
    scorer = ToolReward(tokenizer, [tokenizer.eos_token_id])
    rows, candidates, rejected = prepare_rows(dataset, sources, tokenizer, scorer)
    write_jsonl(data_dir / "rejected.jsonl", rejected)
    save_outputs(data_dir, dataset, revision, sources, rows, candidates, rejected)
    save_reward_examples(rows["train"], scorer, data_dir / "reward_examples.jsonl")
    print(f"Готово: train={len(rows['train'])}, validation={len(rows['validation'])}")
    print(f"ToolACE revision: {revision}. Данные и EDA: {data_dir}")


if __name__ == "__main__":
    main()
