import ast
import json
import re


CONFIG = {
    "tools_marker": "Here is a list of functions in JSON format that you can invoke:",
    "schema_types": {
        "dict": "object", "float": "number", "int": "integer",
        "str": "string", "bool": "boolean", "list": "array",
        "tuple": "array", "any": "string",
    },
}


def convert_schema(value):
    if isinstance(value, list):
        return [convert_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: convert_schema(item) for key, item in value.items()}
    if isinstance(result.get("type"), str):
        result["type"] = CONFIG["schema_types"].get(result["type"], result["type"])
    return result


def extract_tools(system):
    if CONFIG["tools_marker"] not in system:
        raise ValueError("Не найден список инструментов в system")
    tail = system.split(CONFIG["tools_marker"], 1)[1].lstrip()
    functions, _ = json.JSONDecoder().raw_decode(tail)
    if not isinstance(functions, list) or not functions:
        raise ValueError("Пустой или некорректный список инструментов")
    tools = []
    for function in functions:
        if not isinstance(function.get("name"), str):
            raise ValueError("У инструмента нет имени")
        tools.append({"type": "function", "function": {
            "name": function["name"],
            "description": function.get("description", ""),
            "parameters": convert_schema(function["parameters"]),
        }})
    names = [tool["function"]["name"] for tool in tools]
    if len(names) != len(set(names)):
        raise ValueError("Повторяющиеся имена инструментов")
    return tools


def split_calls(text):
    """Запятые внутри строк, аргументов и вложенных списков не разделяют вызовы."""
    parts = []
    start = 0
    depth = 0
    quote = None
    escaped = False
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
        if depth < 0:
            raise ValueError("Незакрытые скобки")
    if quote is not None or depth != 0:
        raise ValueError("Незакрытая строка или скобка")
    if text[start:].strip():
        parts.append(text[start:].strip())
    return parts


def read_literal(node):
    """Python-литералы, а также JSON true/false/null внутри них."""
    if isinstance(node, ast.Name) and node.id in {"true", "false", "null"}:
        return {"true": True, "false": False, "null": None}[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [read_literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {read_literal(key): read_literal(value)
                for key, value in zip(node.keys, node.values, strict=True)}
    return ast.literal_eval(node)


def parse_call(text, tool_names):
    name, separator, arguments = text.partition("(")
    name = name.strip()
    if not separator or not arguments.endswith(")") or name not in tool_names:
        raise ValueError("Некорректный вызов или неизвестное имя функции")
    node = ast.parse("f(" + arguments, mode="eval").body
    if not isinstance(node, ast.Call) or node.args:
        raise ValueError("Поддерживаются только именованные аргументы")
    values = {}
    for keyword in node.keywords:
        if keyword.arg is None or keyword.arg in values:
            raise ValueError("Распаковка или повтор аргумента")
        values[keyword.arg] = read_literal(keyword.value)
    json.dumps(values, allow_nan=False)
    return {"name": name, "arguments": values}


def parse_answer(text, tool_names):
    text = text.strip()
    if text.startswith("["):
        if not text.endswith("]"):
            raise ValueError("Незакрытый список вызовов")
        calls = [parse_call(part, tool_names) for part in split_calls(text[1:-1])]
        if not calls:
            raise ValueError("Пустой список вместо содержательного ответа")
        return calls
    if not text or re.search(r"<tool|<function|```|\w+\([^\n]*=", text):
        raise ValueError("Неоднозначный формат ответа")
    return []


def format_calls(calls):
    """XML-формат Qwen; нужен для истории и просмотра эталонов, не для SFT."""
    blocks = []
    for call in calls:
        lines = ["<tool_call>", f"<function={call['name']}>"]
        for name, value in call["arguments"].items():
            value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            lines.extend([f"<parameter={name}>", value, "</parameter>"])
        lines.extend(["</function>", "</tool_call>"])
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def unfold_dialogue(row):
    tools = extract_tools(row["system"])
    tool_names = {tool["function"]["name"] for tool in tools}
    history = []
    examples = []
    for turn_index, turn in enumerate(row["conversations"]):
        role, content = turn["from"], turn["value"]
        if role not in {"user", "assistant", "tool"}:
            raise ValueError(f"Неизвестная роль: {role}")
        if role == "assistant":
            calls = parse_answer(content, tool_names)
            if history and history[-1]["role"] == "user":
                examples.append({
                    "messages": list(history), "tools": tools, "calls": calls,
                    "reference": format_calls(calls) if calls else content,
                    "turn_index": turn_index,
                })
            content = format_calls(calls) if calls else content
        history.append({"role": role, "content": content})
    return examples
