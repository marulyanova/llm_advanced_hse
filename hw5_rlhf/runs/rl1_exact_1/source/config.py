from pathlib import Path


CONFIG = {
    "root": Path(__file__).resolve().parent,
    "model_id": "Qwen/Qwen3.5-2B",
    "model_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "seed": 42,
    "experiment": "rl1_exact_1",
    "data_dir": "data",
    "runs_dir": "runs",
    "checkpoints_dir": "checkpoints",
    "results_dir": "results",
    "report_dir": "report",
}


def project_path(name):
    return CONFIG["root"] / CONFIG[name]
