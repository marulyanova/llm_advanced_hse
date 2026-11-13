import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText

from config import CONFIG as COMMON_CONFIG, project_path
from utils import package_versions, read_json, write_json


CONFIG = {
    "dtype": "float32",
    "save_dtype": "bfloat16",
    "max_shard_size": "4GB",
    "safe_serialization": True,
}


def main():
    name = COMMON_CONFIG["experiment"]
    run_dir = project_path("runs_dir") / name
    run_config = read_json(run_dir / "config.json")
    selection = read_json(run_dir / "selection.json")
    output_dir = project_path("checkpoints_dir") / name
    if output_dir.exists():
        raise FileExistsError(f"Папка {output_dir} уже существует")
    base_config = run_config["common"]
    model = AutoModelForImageTextToText.from_pretrained(
        base_config["model_id"], revision=base_config["model_revision"],
        dtype=getattr(torch, CONFIG["dtype"]), device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, run_dir / selection["adapter"])
    model = model.merge_and_unload(safe_merge=True)
    model = model.to(dtype=getattr(torch, CONFIG["save_dtype"]))
    model.config.get_text_config().use_cache = True
    model.save_pretrained(
        output_dir, safe_serialization=CONFIG["safe_serialization"],
        max_shard_size=CONFIG["max_shard_size"],
    )
    shutil.copy2(run_dir / "config.json", output_dir / "training_config.json")
    write_json(output_dir / "export.json", {
        "config": CONFIG, "selection": selection, "versions": package_versions(),
        "base_model": base_config["model_id"], "base_revision": base_config["model_revision"],
    })
    print(f"Полный Hugging Face чекпоинт: {output_dir}")
    print(f"python3 evaluate.py ./checkpoints/{name} --output results/{name}")


if __name__ == "__main__":
    main()
