"""Подсчёт метрик MASSIVE и проверка сохранённых результатов на dev."""

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def normalize_value(value: str) -> str:
    """Приводит значение slot к виду для сравнения."""
    return " ".join(value.casefold().split())

def parse_answer(
    text: str, intents: Sequence[str], slot_types: Sequence[str],
) -> dict[str, Any] | None:
    """Разбирает ответ модели и проверяет формат.

    Args:
        text: сырой ответ модели.
        intents: допустимые названия intent.
        slot_types: допустимые типы slots.

    Returns:
        Словарь с ключами `intent` и `slots` или None, если ответ невалиден.
    """
    try:
        obj = json.loads(text.strip())
        if not isinstance(obj, dict) or set(obj) != {"intent", "slots"}:
            return None
        if not isinstance(obj["intent"], str) or obj["intent"] not in intents:
            return None
        if not isinstance(obj["slots"], list):
            return None
        for slot in obj["slots"]:
            if not isinstance(slot, list) or len(slot) != 2:
                return None
            kind, value = slot
            if (not isinstance(kind, str) or kind not in slot_types
                    or not isinstance(value, str) or not value.strip()):
                return None
        return obj
    except (ValueError, TypeError):
        return None

def slot_counter(obj: dict[str, Any]) -> Counter:
    """Превращает slots ответа в мультимножество

    Args:
        obj: разобранный ответ с ключом `slots`.

    Returns:
        Counter пар `(тип, нормализованное значение)`.
    """
    return Counter((kind, normalize_value(value)) for kind, value in obj["slots"])

def compute_metrics(
    predictions: Sequence[str],
    golds: Sequence[str],
    intents: Sequence[str],
    slot_types: Sequence[str],
) -> dict[str, float]:
    """Считает метрики качества по ответам модели.

    Args:
        predictions: ответы модели.
        golds: эталонные JSON в том же порядке.
        intents: допустимые названия intent.
        slot_types: допустимые типы slots.

    Returns:
        Словарь с `quality`, `intent_accuracy`, `slot_micro_f1`,
        `frame_exact_match`, `valid_json_rate` и числом примеров `n`.
    """
    if len(predictions) != len(golds) or not golds:
        raise ValueError("Нужны непустые списки одинаковой длины.")
    correct_intent = exact = valid = tp = fp = fn = 0
    for text, gold_text in zip(predictions, golds):
        gold = parse_answer(gold_text, intents, slot_types)
        if gold is None:
            raise ValueError("Невалидная gold-разметка")
        pred = parse_answer(text, intents, slot_types)
        gold_slots = slot_counter(gold)
        pred_slots = Counter() if pred is None else slot_counter(pred)
        tp += sum((pred_slots & gold_slots).values())
        fp += sum((pred_slots - gold_slots).values())
        fn += sum((gold_slots - pred_slots).values())
        if pred is not None:
            valid += 1
            intent_ok = pred["intent"] == gold["intent"]
            correct_intent += int(intent_ok)
            exact += int(intent_ok and pred_slots == gold_slots)
    n = len(golds)
    denom = 2 * tp + fp + fn
    slot_f1 = 2 * tp / denom if denom else float(valid == n)
    intent_acc = correct_intent / n
    return {
        "quality": 0.5 * (intent_acc + slot_f1),
        "intent_accuracy": intent_acc,
        "slot_micro_f1": slot_f1,
        "frame_exact_match": exact / n,
        "valid_json_rate": valid / n,
        "n": n,
    }

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def recompute_saved_metrics(artifacts: Path) -> dict[str, dict[str, float]]:
    """Сверяет одни и те же примеры dev и пересчитывает метрики из сырых ответов."""
    schema = read_json(artifacts / "massive_schema.json")
    dev = artifacts / "dev"
    reference_rows = read_jsonl(dev / "baseline_dev_predictions.jsonl")
    expected_examples = [
        (row["id"], row["text"], row["gold"]) for row in reference_rows
    ]
    if len(expected_examples) != 256 or len({row[0] for row in expected_examples}) != 256:
        raise ValueError("Ожидаются 256 уникальных примеров dev.")
    golds = [row["gold"] for row in reference_rows]
    results = {}
    for path in sorted(dev.glob("*_dev.json")):
        run = path.name.removesuffix("_dev.json")
        rows = read_jsonl(dev / f"{run}_dev_predictions.jsonl")
        examples = [(row["id"], row["text"], row["gold"]) for row in rows]
        if examples != expected_examples:
            raise ValueError(f"{run}: сохранённые примеры dev отличаются от бейзлайна.")
        predictions = [row["prediction"] for row in rows]
        metrics = compute_metrics(predictions, golds, schema["intents"], schema["slot_types"])
        if metrics != read_json(path):
            raise ValueError(f"{run}: пересчитанные метрики расходятся с {path.name}.")
        results[run] = metrics
    return results


def print_saved_metrics(artifacts: Path) -> None:
    results = recompute_saved_metrics(artifacts)
    metric_names = ("quality", "intent_accuracy", "slot_micro_f1", "frame_exact_match", "valid_json_rate")
    print("run", *metric_names, "n", sep="\t")
    for run, metrics in results.items():
        print(run, *(f"{metrics[name]:.4f}" for name in metric_names), metrics["n"], sep="\t")


if __name__ == "__main__":
    print_saved_metrics(Path(__file__).resolve().parent / "artifacts")
