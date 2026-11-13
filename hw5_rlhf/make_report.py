import csv
import json
from collections import Counter
from html import escape

from config import CONFIG as COMMON_CONFIG, project_path
from utils import read_json, read_jsonl, write_jsonl


CONFIG = {
    "runs": [
        "base", "rl1_exact", "rl2_dense", "rl3_dr_grpo", "rl4_dapo",
        "rl5_no_kl", "rl6_low_temperature", "rl7_learning_rate", "rl8_no_reward_scaling",
        "rl9_gspo", "rl10_cispo",
    ],
    "groups": ["simple", "multiple", "parallel", "parallel_multiple", "irrelevance"],
    "error_examples_per_category": 2,
    "reward_examples_limit": 12,
    "chart_width": 700,
    "chart_height": 230,
    "chart_margin": 45,
    "author": "Укажите имя",
    "conclusions": {
        "Влияние изменений": "TODO: сравните RL-1 с RL-2; RL-2 с RL-4–RL-10; "
                              "RL-8 с RL-3 по overall и категориям.",
        "Награда и BFCL": "TODO: сопоставьте кривые exact/reward и BFCL выбранных шагов. "
                            "Одной оценки BFCL на запуск недостаточно для корреляции во времени.",
        "Оставшиеся ошибки": "TODO: разберите приведённые ниже примеры, включая лишний "
                              "и пропущенный вызов; объясните причины и возможные улучшения.",
    },
}

STYLE = """
body {font: 15px/1.55 sans-serif; color: #20252b; max-width: 1000px; margin: 36px auto; padding: 20px;}
h1, h2, h3 {line-height: 1.2; color: #173c58;}
table {border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px;}
td, th {border: 1px solid #c9d1d9; padding: 7px; text-align: left;}
pre {white-space: pre-wrap; overflow-wrap: anywhere; background: #f4f6f8; padding: 12px; font-size: 11px;}
svg {width: 100%; height: auto; display: block;}
figure {margin: 18px 0; break-inside: avoid;}
.error {border-left: 3px solid #c68436; padding-left: 15px; margin: 20px 0;}
@media print {body {margin: 0; padding: 0; font-size: 11px;} h2, h3 {break-after: avoid;}}
@page {size: A4; margin: 15mm;}
"""


def paragraph(text):
    return f"<p>{escape(str(text))}</p>"


def json_block(value):
    return "<pre>" + escape(json.dumps(value, ensure_ascii=False, indent=2)) + "</pre>"


def table(headers, rows):
    head = "".join(f"<th>{escape(str(item))}</th>" for item in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{escape(str(item))}</td>" for item in row)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def chart(points, title, x_label, bars=False):
    if not points:
        return paragraph(f"{title}: данных пока нет.")
    width, height, margin = CONFIG["chart_width"], CONFIG["chart_height"], CONFIG["chart_margin"]
    xmin, xmax = min(x for x, y in points), max(x for x, y in points)
    ymin, ymax = min(0, min(y for x, y in points)), max(y for x, y in points)
    dx, dy = max(xmax - xmin, 1), max(ymax - ymin, 1e-9)
    coordinates = [(margin + (x - xmin) / dx * (width - 2 * margin),
                    height - margin - (y - ymin) / dy * (height - 2 * margin))
                   for x, y in points]
    elements = [f'<path d="M {margin} {margin} V {height-margin} H {width-margin}" '
                'fill="none" stroke="#8899a6"/>']
    if bars:
        for x, y in coordinates:
            elements.append(f'<path d="M {x:.1f} {height-margin} V {y:.1f}" '
                            'stroke="#267a9e" stroke-width="10"/>')
    else:
        line = " ".join(f"{x:.1f},{y:.1f}" for x, y in coordinates)
        elements.append(f'<polyline points="{line}" fill="none" stroke="#267a9e" stroke-width="2"/>')
        for x, y in coordinates:
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" fill="#267a9e"/>')
    for x, y, text in [(5, margin, f"{ymax:.3g}"), (5, height-margin, f"{ymin:.3g}"),
                       (margin, height-12, str(xmin)), (width-margin, height-12, str(xmax))]:
        elements.append(f'<text x="{x}" y="{y}" font-size="11">{escape(text)}</text>')
    return (f'<figure><figcaption>{escape(title)}; X: {escape(x_label)}</figcaption>'
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'role="img" aria-label="{escape(title)}">'
            + "".join(elements) + "</svg></figure>")


def load_results():
    results = {}
    signature = None
    for run in CONFIG["runs"]:
        path = project_path("results_dir") / run / "metrics.json"
        if not path.exists():
            continue
        metrics = read_json(path)
        current = (metrics["benchmark_sha256"], metrics["base_revision"], metrics["decoding"])
        if signature is not None and current != signature:
            raise ValueError("У запусков отличаются бенчмарк, ревизия или параметры оценки")
        signature = current
        results[run] = metrics
    return results


def benchmark_table(results, output_dir):
    rows = []
    for run in CONFIG["runs"]:
        metrics = results.get(run)
        if metrics is None:
            rows.append([run] + ["не запущено"] * (len(CONFIG["groups"]) + 1))
        else:
            scores = [metrics["overall"]] + [metrics["groups"][key] for key in CONFIG["groups"]]
            rows.append([run] + [f"{score:.2%}" for score in scores])
    headers = ["Запуск", "Overall"] + CONFIG["groups"]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    return "<h2>BFCL</h2>" + table(headers, rows) + paragraph(
        "Overall перенесён из evaluate.py без пересчёта: это среднее точностей пяти групп, "
        "а не доля правильных ответов среди всех 350 примеров."
    )


def data_section():
    path = project_path("data_dir") / "eda.json"
    if not path.exists():
        return "<h2>Данные</h2>" + paragraph("Сначала выполните prepare_data.py.")
    eda = read_json(path)
    manifest = read_json(project_path("data_dir") / "manifest.json")
    parts = ["<h2>Данные</h2>", paragraph(
        f"ToolACE revision: {manifest['dataset_revision']}. Seed: {manifest['common_config']['seed']}. "
        f"Исходных диалогов: {eda['raw_dialogues']}; дубликатов: {eda['duplicate_dialogues']}."
    ), paragraph(
        "Точные дубликаты диалогов удалены до разделения. 90% диалогов идут в train, 10% в validation "
        "при настройках по умолчанию. После разделения из каждого диалога взяты префиксы перед "
        "ответами ассистента на сообщения пользователя. Результаты инструментов остаются в истории. "
        "Длинные примеры и неподдерживаемые эталоны отфильтрованы, промпты не обрезались."
    ), json_block({"filters": manifest["config"], "rejected": eda["rejected"]})]
    for split, stats in eda["after_filtering"].items():
        parts.append(f"<h3>{escape(split)}</h3>")
        parts.append(json_block({key: stats[key] for key in (
            "examples", "dialogues", "kinds", "calls_per_answer", "unique_tool_names", "top_tool_names"
        )}))
        for field in ("prompt_tokens", "reference_tokens", "tools_per_prompt"):
            summary = stats[field]
            parts.append(json_block({field: {key: value for key, value in summary.items() if key != "histogram"}}))
            points = [(bucket["left"], bucket["count"]) for bucket in summary.get("histogram", [])]
            parts.append(chart(points, f"{split}: {field}, число примеров", field, bars=True))
    return "".join(parts)


def reward_section():
    parts = ["<h2>Награда</h2>", paragraph(
        "Точная награда R_exact = I[валидный ответ и совпало мультимножество вызовов]. "
        "Порядок вызовов и ключей словаря не важен, повторы вызовов учитываются. "
        "При ожидаемом вызове частичная награда R_dense = w_exact E + w_name F_name + w_arg F_arg; "
        "по умолчанию веса 0.6, 0.2, 0.2. F — multiset F1: 2 × число совпадений / "
        "(число предсказанных + число эталонных элементов). F_name сравнивает имена, "
        "F_arg — тройки (имя функции, имя аргумента, значение). Для двух пустых наборов F = 1."
    ), paragraph(
        "При отсутствии ожидаемого вызова обе награды равны 1 за непустой корректный текст без вызовов. "
        "Пустые, незавершённые, некорректно размеченные ответы и недоступные функции получают 0. "
        "Если вызов требовался, ответ без вызовов получает 0. Частичная награда даёт более частый сигнал, "
        "но может поощрять перенос аргументов между вызовами одной функции. Только точная компонента "
        "проверяет их совместную принадлежность. Смысл текстового ответа без вызова не оценивается. "
        "ToolACE exact строже BFCL: в нём нет альтернативных допустимых значений и правил AST-checker."
    )]
    path = project_path("data_dir") / "reward_examples.jsonl"
    if path.exists():
        for example in read_jsonl(path)[:CONFIG["reward_examples_limit"]]:
            parts.append(json_block(example))
    return "".join(parts)


def run_section(name, output_dir):
    run_dir = project_path("runs_dir") / name
    if not (run_dir / "config.json").exists():
        return f"<h3>{escape(name)}</h3>" + paragraph("Обучение не запускалось.")
    config = read_json(run_dir / "config.json")
    selection = read_json(run_dir / "selection.json") if (run_dir / "selection.json").exists() else {}
    parts = [f"<h3>{escape(name)}</h3>", json_block({
        "experiment": config["experiment"], "training": config["training"],
        "reward": config["reward"], "selection": selection,
        "validation_kinds": config["validation_kinds"],
    })]
    logs_path = run_dir / "logs.jsonl"
    logs = read_jsonl(logs_path) if logs_path.exists() else []
    for field in ("loss", "reward", "quality/exact", "eval_quality/exact", "frac_reward_zero_std"):
        points = [(row["step"], row[field]) for row in logs if isinstance(row.get(field), (int, float))]
        figure = chart(points, f"{name}: {field}", "шаг оптимизатора")
        parts.append(figure)
        if points:
            svg = figure[figure.index("<svg"):figure.index("</svg>") + len("</svg>")]
            (output_dir / f"{name}_{field.replace('/', '_')}.svg").write_text(svg, encoding="utf-8")
    return "".join(parts)


def error_type(prediction):
    if prediction["finish_reason"] == "length":
        return "обрезанный ответ"
    if "irrelevance" in prediction["category"] and prediction["calls"]:
        return "лишний вызов"
    if "irrelevance" not in prediction["category"] and not prediction["calls"]:
        return "пропущенный вызов"
    return "неверная функция, аргументы или разметка"


def collect_errors(name, benchmark):
    path = project_path("results_dir") / name / "predictions.jsonl"
    if not path.exists():
        return []
    errors = []
    for prediction in read_jsonl(path):
        if prediction["check"]["valid"]:
            continue
        source = benchmark[prediction["id"]]
        errors.append({
            **prediction, "error_type": error_type(prediction),
            "question": source["question"], "function": source["function"],
            "ground_truth": source["ground_truth"],
        })
    return errors


def errors_section(results, output_dir):
    trained = [name for name in CONFIG["runs"] if name != "base" and name in results]
    if not trained:
        return "<h2>Ошибки</h2>" + paragraph("Нет результатов обученных моделей.")
    best = max(trained, key=lambda name: results[name]["overall"])
    if not (project_path("results_dir") / best / "predictions.jsonl").exists():
        return "<h2>Ошибки</h2>" + paragraph(f"Для {best} отсутствует predictions.jsonl.")
    benchmark = {row["id"]: row for row in read_jsonl(COMMON_CONFIG["root"] / "bfcl.jsonl")}
    errors = collect_errors(best, benchmark)
    write_jsonl(output_dir / "best_model_errors.jsonl", errors)
    parts = ["<h2>Ошибки лучшей из оценённых RL-моделей</h2>", paragraph(best),
             json_block(dict(Counter(row["error_type"] for row in errors)))]
    shown = Counter()
    for error in errors:
        group = (error["category"], error["error_type"])
        if shown[group] >= CONFIG["error_examples_per_category"]:
            continue
        shown[group] += 1
        parts.append('<div class="error">' + json_block(error) + "</div>")
    if not errors:
        parts.append(paragraph("Сохранённых ошибок нет; проверьте полноту predictions.jsonl."))
    return "".join(parts)


def conclusions_section():
    parts = ["<h2>Выводы и ограничения</h2>", paragraph(
        "RL-1 и RL-2 отличаются только наградой; RL-2 и RL-3 — нормировками Dr. GRPO. "
        "Относительно RL-2 в RL-5 отключён KL, в RL-6 снижена температура, "
        "в RL-7 увеличен learning rate, в RL-8 отключено масштабирование advantage. "
        "RL-8 и RL-3 отличаются нормировкой loss. RL-4 меняет нормировку loss и верхний clipping; "
        "dynamic sampling и overlong reward shaping в этом запуске не реализованы. "
        "RL-9 относительно RL-2 меняет только token ratio на sequence ratio (GSPO), "
        "сохраняя границы clipping. RL-10 использует CISPO: отсоединённый вес "
        "min(ratio, 5.0) умножает advantage и log-probability; loss усредняется по токенам. "
        "Это контролируемые сравнения при неизменных данных и остальных параметрах. "
        "Один seed не позволяет оценить разброс между запусками. "
        "Чекпоинт выбирается по точной награде на validation ToolACE с sampling, а не по BFCL."
    ), paragraph(
        "Если все ответы на запрос получают одинаковую награду, R_i − mean(R) = 0. "
        "Advantage равен нулю и в GRPO с нормировкой, и в Dr. GRPO. "
        "Обучающего сигнала от награды нет, но KL-штраф при beta > 0 может давать градиент."
    )]
    for title, text in CONFIG["conclusions"].items():
        parts.extend([f"<h3>{escape(title)}</h3>", paragraph(text)])
    return "".join(parts)


def main():
    output_dir = project_path("report_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results()
    parts = ["<h1>RL для вызова инструментов</h1>", paragraph(CONFIG["author"]),
             paragraph("Черновик отчёта. Заполните выводы после анализа фактических результатов."),
             benchmark_table(results, output_dir), data_section(), reward_section(), "<h2>Запуски</h2>"]
    for name in CONFIG["runs"]:
        if name != "base":
            parts.append(run_section(name, output_dir))
    parts.extend([errors_section(results, output_dir), conclusions_section()])
    document = ('<!doctype html><html lang="ru"><head><meta charset="utf-8">'
                f"<title>RL Homework</title><style>{STYLE}</style></head><body>"
                + "".join(parts) + "</body></html>")
    (output_dir / "report.html").write_text(document, encoding="utf-8")
    print(f"Отчёт: {output_dir / 'report.html'}; таблица: {output_dir / 'summary.csv'}")
    print("Откройте HTML в браузере и сохраните через Печать → Сохранить как PDF.")


if __name__ == "__main__":
    main()
