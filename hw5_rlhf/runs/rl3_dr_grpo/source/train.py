import json
import os
import shutil
from collections import Counter

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoTokenizer, TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer

from config import CONFIG as COMMON_CONFIG, project_path
from rewards import CONFIG as REWARD_CONFIG, ToolReward
from utils import file_hash, package_versions, read_json, read_jsonl, write_json, write_jsonl


CONFIG = {
    "max_steps": 10,
    "learning_rate": 5e-6,
    "lr_scheduler_type": "constant_with_warmup",
    "warmup_steps": 1,
    "max_grad_norm": 1.0,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "per_device_eval_batch_size": 16,
    "num_generations": 4,
    "num_iterations": 1,
    "max_completion_length": 1024,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 0,
    "beta": 0.01,
    "epsilon": 0.2,
    "epsilon_high": 0.2,
    "logging_steps": 5,
    "save_steps": 10,
    "eval_steps": 10,
    "save_total_limit": 2,
    "validation_examples": 8,
    "max_train_examples": None,
    "metric_for_best_model": "eval_quality/exact",
    "gradient_checkpointing": False,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "bf16": True,
    "model_dtype": "bfloat16",
    "optim": "adamw_torch",
    "attn_implementation": "sdpa",
    "use_vllm": False,
    "mask_truncated_completions": True,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "lora_bias": "none",
    "lora_task_type": "CAUSAL_LM",
    "importance_sampling_level": "token",
    "report_to": "none",
    "remove_unused_columns": False,
    "save_strategy": "steps",
    "eval_strategy": "steps",
    "load_best_model_at_end": True,
    "greater_is_better": True,
    "logging_first_step": True,
    "lora_target_modules": (
        r"model\.language_model\..*\."
        r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|"
        r"in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)"
    ),
}

EXPERIMENTS = {
    "rl1_exact": {
        "reward": "exact", "loss_type": "grpo", "scale_rewards": "group",
        "hypothesis": "Точная награда улучшит выбор инструментов и полный набор аргументов.",
    },
    "rl2_dense": {
        "reward": "dense", "loss_type": "grpo", "scale_rewards": "group",
        "hypothesis": "Частичная награда даст сигнал, когда точных ответов в группе ещё нет.",
    },
    "rl3_dr_grpo": {
        "reward": "dense", "loss_type": "dr_grpo", "scale_rewards": "none",
        "hypothesis": "Нормировки Dr. GRPO уменьшат смещение по длине и сложности запроса.",
    },
    "rl4_dapo": {
        "reward": "dense", "loss_type": "dapo", "scale_rewards": "group",
        "overrides": {"epsilon_high": 0.28},
        "hypothesis": "Нормировка по токенам и более широкий верхний clipping улучшат несколько вызовов.",
    },
    "rl5_no_kl": {
        "reward": "dense", "loss_type": "grpo", "scale_rewards": "group",
        "overrides": {"beta": 0.0},
        "hypothesis": "Отсутствие KL позволит быстрее менять поведение, но может ухудшить ответы без вызова.",
    },
    "rl6_low_temperature": {
        "reward": "dense", "loss_type": "grpo", "scale_rewards": "group",
        "overrides": {"temperature": 0.5},
        "hypothesis": "Более низкая температура уменьшит ошибки формата, но сократит разнообразие группы.",
    },
    "rl7_learning_rate": {
        "reward": "dense", "loss_type": "grpo", "scale_rewards": "group",
        "overrides": {"learning_rate": 1e-5},
        "hypothesis": "Увеличение learning rate ускорит рост качества при том же числе шагов.",
    },
    "rl8_no_reward_scaling": {
        "reward": "dense", "loss_type": "grpo", "scale_rewards": "none",
        "hypothesis": "Отключение деления advantage на std уменьшит усиление малых различий награды.",
    },
    "rl9_gspo": {
        "reward": "dense", "loss_type": "grpo", "scale_rewards": "group",
        "overrides": {"importance_sampling_level": "sequence"},
        "hypothesis": "Отношение вероятностей на уровне ответа стабилизирует обновления для нескольких вызовов.",
    },
    "rl10_cispo": {
        "reward": "dense", "loss_type": "cispo", "scale_rewards": "group",
        "overrides": {"epsilon_high": 5.0},
        "hypothesis": "Ограниченный отсоединённый importance weight сохранит сигнал для токенов за границей clipping.",
    },
}


class JsonLogger(TrainerCallback):
    def __init__(self, path):
        self.path = path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": state.global_step, **logs}) + "\n")


def load_data():
    data_dir = project_path("data_dir")
    manifest = read_json(data_dir / "manifest.json")
    for name, expected_hash in manifest["files"].items():
        if file_hash(data_dir / name) != expected_hash:
            raise ValueError(f"Файл {name} изменился после подготовки данных")
    base = manifest["common_config"]
    for key in ("model_id", "model_revision", "seed"):
        if base[key] != COMMON_CONFIG[key]:
            raise ValueError(f"{key} не совпадает с настройками подготовки данных")
    train = Dataset.from_list(read_jsonl(data_dir / "train.jsonl"))
    validation = Dataset.from_list(read_jsonl(data_dir / "validation.jsonl"))
    if set(train["dialogue_id"]) & set(validation["dialogue_id"]):
        raise ValueError("Один диалог попал в оба сплита")
    train = train.shuffle(seed=COMMON_CONFIG["seed"])
    validation = validation.shuffle(seed=COMMON_CONFIG["seed"])
    if CONFIG["max_train_examples"] is not None:
        train = train.select(range(min(len(train), CONFIG["max_train_examples"])))
    validation = validation.select(range(min(len(validation), CONFIG["validation_examples"])))
    return train, validation, manifest


def training_arguments(output_dir, experiment):
    keys = [
        "max_steps", "learning_rate", "lr_scheduler_type", "warmup_steps", "max_grad_norm",
        "per_device_train_batch_size", "gradient_accumulation_steps", "per_device_eval_batch_size",
        "num_generations", "num_iterations", "max_completion_length", "temperature", "top_p", "top_k",
        "beta", "epsilon", "epsilon_high", "logging_steps", "save_steps", "eval_steps",
        "save_total_limit", "metric_for_best_model", "gradient_checkpointing",
        "gradient_checkpointing_kwargs", "bf16", "optim", "use_vllm", "mask_truncated_completions",
        "importance_sampling_level", "report_to", "remove_unused_columns", "save_strategy",
        "eval_strategy", "load_best_model_at_end", "greater_is_better", "logging_first_step",
    ]
    settings = {key: CONFIG[key] for key in keys}
    settings.update(experiment.get("overrides", {}))
    return GRPOConfig(
        output_dir=str(output_dir), **settings,
        loss_type=experiment["loss_type"], scale_rewards=experiment["scale_rewards"],
        seed=COMMON_CONFIG["seed"], data_seed=COMMON_CONFIG["seed"],
    )


def make_trainer(train, validation, output_dir, experiment):
    tokenizer = AutoTokenizer.from_pretrained(
        COMMON_CONFIG["model_id"], revision=COMMON_CONFIG["model_revision"],
        padding_side="left",
    )
    model = AutoModelForImageTextToText.from_pretrained(
        COMMON_CONFIG["model_id"], revision=COMMON_CONFIG["model_revision"],
        dtype=getattr(torch, CONFIG["model_dtype"]), attn_implementation=CONFIG["attn_implementation"],
    )
    eos_ids = model.generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    eos_ids = [token for token in eos_ids if token is not None] + [tokenizer.eos_token_id]
    scorer = ToolReward(tokenizer, eos_ids)
    adapter = LoraConfig(
        r=CONFIG["lora_rank"], lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"], target_modules=CONFIG["lora_target_modules"],
        bias=CONFIG["lora_bias"], task_type=CONFIG["lora_task_type"],
    )
    return GRPOTrainer(
        model=model, processing_class=tokenizer, peft_config=adapter,
        args=training_arguments(output_dir, experiment),
        reward_funcs=scorer.make_function(experiment["reward"]),
        train_dataset=train, eval_dataset=validation,
        callbacks=[JsonLogger(output_dir / "logs.jsonl")],
    )


def save_run_config(output_dir, experiment, train, validation, manifest):
    write_json(output_dir / "config.json", {
        "common": COMMON_CONFIG,
        "training": {**CONFIG, **experiment.get("overrides", {})}, "experiment": experiment,
        "reward": REWARD_CONFIG, "data_manifest": manifest,
        "versions": package_versions(), "gpu": torch.cuda.get_device_name(0),
        "train_examples": len(train), "validation_examples": len(validation),
        "train_kinds": dict(Counter(train["kind"])),
        "validation_kinds": dict(Counter(validation["kind"])),
    })
    write_jsonl(output_dir / "validation_ids.jsonl", [{"id": item} for item in validation["id"]])
    source_dir = output_dir / "source"
    source_dir.mkdir()
    for name in ("config.py", "train.py", "rewards.py", "toolace.py", "utils.py", "prepare_data.py"):
        shutil.copy2(COMMON_CONFIG["root"] / name, source_dir / name)


def save_selection(trainer, output_dir):
    checkpoint = trainer.state.best_model_checkpoint
    if checkpoint is None:
        raise RuntimeError("Нет выбранного чекпоинта. Проверьте max_steps и eval_steps.")
    trainer.save_model(str(output_dir / "best_adapter"))
    trainer.save_state()
    write_json(output_dir / "selection.json", {
        "criterion": CONFIG["metric_for_best_model"],
        "best_metric": trainer.state.best_metric,
        "checkpoint": checkpoint,
        "selected_step": int(checkpoint.rsplit("-", 1)[1]),
        "total_steps": trainer.state.global_step,
        "adapter": "best_adapter",
    })


def main():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Нужна CUDA GPU с поддержкой bfloat16.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() != 1:
        raise RuntimeError("Скрипт рассчитан на одну GPU. Ограничьте CUDA_VISIBLE_DEVICES.")
    name = COMMON_CONFIG["experiment"]
    experiment = EXPERIMENTS[name]
    output_dir = project_path("runs_dir") / name
    if output_dir.exists():
        raise FileExistsError(f"Запуск {output_dir} уже есть. Выберите новое имя эксперимента.")
    if CONFIG["max_steps"] < CONFIG["eval_steps"]:
        raise ValueError("max_steps должен быть не меньше eval_steps")
    set_seed(COMMON_CONFIG["seed"])
    train, validation, manifest = load_data()
    output_dir.mkdir(parents=True)
    save_run_config(output_dir, experiment, train, validation, manifest)
    trainer = make_trainer(train, validation, output_dir, experiment)
    write_json(output_dir / "trainer_args.json", trainer.args.to_dict())
    trainer.train()
    save_selection(trainer, output_dir)
    print(f"Выбранный LoRA-адаптер: {output_dir / 'best_adapter'}")


if __name__ == "__main__":
    main()
