import json
from collections import Counter
from statistics import mean

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

from config import CONFIG as COMMON_CONFIG


CONFIG = {
    "exact_weight": 0.6,
    "name_weight": 0.2,
    "argument_weight": 0.2,
    "tag_pairs": [
        ("<tool_call>", "</tool_call>"),
        ("<function=", "</function>"),
        ("<parameter=", "</parameter>"),
    ],
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def multiset_f1(predicted, expected):
    predicted, expected = Counter(predicted), Counter(expected)
    size = sum(predicted.values()) + sum(expected.values())
    if size == 0:
        return 1.0
    matched = sum((predicted & expected).values())
    return 2.0 * matched / size


def argument_items(calls):
    items = []
    for call in calls:
        for name, value in call["arguments"].items():
            items.append((call["name"], name, canonical(value)))
    return items


def compare_calls(predicted, expected):
    exact = Counter(map(canonical, predicted)) == Counter(map(canonical, expected))
    names = multiset_f1(
        [call["name"] for call in predicted],
        [call["name"] for call in expected],
    )
    arguments = multiset_f1(argument_items(predicted), argument_items(expected))
    return {"exact": float(exact), "names": names, "arguments": arguments}


class ToolReward:
    def __init__(self, tokenizer, eos_token_ids):
        self.tokenizer = tokenizer
        self.eos_token_ids = set(eos_token_ids)
        self.parser = Qwen3CoderToolParser(tokenizer)

    def parse(self, text, tools):
        request = ChatCompletionRequest.model_validate({
            "model": COMMON_CONFIG["model_id"],
            "messages": [{"role": "user", "content": "Choose the appropriate tools."}],
            "tools": tools,
        })
        self.parser.tools = request.tools
        parsed = self.parser.extract_tool_calls(text, request)
        calls = [{"name": call.function.name,
                  "arguments": json.loads(call.function.arguments)}
                 for call in parsed.tool_calls]
        if any(text.count(left) != text.count(right)
               for left, right in CONFIG["tag_pairs"]):
            raise ValueError("Несбалансированные теги")
        if text.count("<tool_call>") != len(calls):
            raise ValueError("Не все вызовы разобраны")
        if text.count("<function=") != len(calls):
            raise ValueError("Неверное число функций")
        if not calls and any(tag in text for tag in ("<tool", "<function", "<parameter")):
            raise ValueError("Некорректная разметка вызова")
        known_names = {tool["function"]["name"] for tool in tools}
        if any(call["name"] not in known_names for call in calls):
            raise ValueError("Вызвана недоступная функция")
        if text.count("<parameter=") != sum(len(call["arguments"]) for call in calls):
            raise ValueError("Повторяющиеся или неразобранные параметры")
        canonical(calls)
        return calls

    def score(self, text, tools, expected, truncated=False):
        result = {"valid": 0.0, "exact": 0.0, "names": 0.0,
                  "arguments": 0.0, "dense": 0.0}
        if truncated or not text.strip():
            return result
        try:
            calls = self.parse(text, tools)
        except (ValueError, TypeError, KeyError):
            return result
        if not calls and text.lstrip().startswith(("[", "{", "```")):
            return result
        result["valid"] = 1.0
        if not expected:
            correct = float(not calls)
            result.update(exact=correct, names=correct, arguments=correct, dense=correct)
        elif calls:
            result.update(compare_calls(calls, expected))
            result["dense"] = (
                CONFIG["exact_weight"] * result["exact"]
                + CONFIG["name_weight"] * result["names"]
                + CONFIG["argument_weight"] * result["arguments"]
            )
        return result

    def decode(self, ids):
        truncated = not ids or ids[-1] not in self.eos_token_ids
        end = len(ids)
        while end and ids[end - 1] in self.eos_token_ids:
            end -= 1
        text = self.tokenizer.decode(ids[:end], skip_special_tokens=False)
        return text, truncated

    def make_function(self, reward_type):
        if reward_type not in {"exact", "dense"}:
            raise ValueError(f"Неизвестная награда: {reward_type}")

        def reward(completion_ids, tools_json, expected_json, log_metric=None, **kwargs):
            scores = []
            for ids, tools, expected in zip(
                completion_ids, tools_json, expected_json, strict=True
            ):
                text, truncated = self.decode(ids)
                scores.append(self.score(
                    text, json.loads(tools), json.loads(expected), truncated
                ))
            if log_metric is not None:
                for name in ("valid", "exact", "names", "arguments"):
                    log_metric(f"quality/{name}", mean(score[name] for score in scores))
            return [score[reward_type] for score in scores]

        reward.__name__ = f"tool_{reward_type}"
        return reward
