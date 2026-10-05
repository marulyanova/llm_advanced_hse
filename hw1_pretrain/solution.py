import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
import torch
import time

# Don't change this parameter
MAX_TRAINING_TIME_SECONDS = 60 * 15
MAX_LENGTH = 512
INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
LABELS = "labels"

# Don't change these parameters
TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = "./output_dir"
NUM_SHARDS = 32
VALIDATION_SIZE = 5000


# TODO: Configure training parameters
TRAINING_CONFIG = {
    "output_dir": f"{OUTPUT_DIR}/gpt2-1b-russian",
    "optim": "adamw_torch_fused",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "save_steps": 100,
    "save_total_limit": 2,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "warmup_steps": 50,
    "logging_steps": 1,
    "eval_steps": 100,
    "eval_strategy": "steps",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "bf16": False,
    "tf32": True,
    "gradient_checkpointing": False,
    "gradient_accumulation_steps": 8,
    "dataloader_num_workers": 4,
    "torch_compile": False,
    "report_to": "none",
}


class TimeoutCallback(TrainerCallback):
    """Callback to stop training after a specified timeout."""

    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                control.should_training_stop = True
                # Include the final weights in best-checkpoint selection.
                control.should_evaluate = True
                control.should_save = True
                print(f"Training stopped after {elapsed:.2f} seconds")
        return control


def prepare_tokenizer():
    """
    TODO: Implement tokenizer preparation.
    - Load the tokenizer from TOKENIZER_NAME
    - Set pad_token to eos_token
    - Return the tokenizer
    """

    tokenizer = AutoTokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def tokenize_function(examples, tokenizer):
    """
    TODO: Implement tokenization function.
    - Tokenize the text with truncation and padding to MAX_LENGTH
    - Create labels from input_ids
    - Return dictionary with 'labels', 'input_ids', and 'attention_mask'
    """

    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def save_as_parquets(ds, output_dir=OUTPUT_DIR, num_shards=NUM_SHARDS):
    """
    TODO: Implement saving dataset as parquet shards.
    - Create output directory if it doesn't exist
    - Split dataset into num_shards shards
    - Save each shard as a parquet file with format: {output_dir}/{index:05d}.parquet
    """

    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=i, contiguous=True)
        shard.to_parquet(f"{output_dir}/{i:05d}.parquet")


def prepare_dataset():
    """
    TODO: Implement dataset preparation.
    - Load the Wikipedia dataset: "wikimedia/wikipedia", "20231101.ru", split="train"
    - Tokenize the dataset using tokenize_function
    - Save as parquet files
    """
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train[:50000]")
    tokenizer = prepare_tokenizer()

    tokenized = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        num_proc=os.cpu_count(),
    )

    save_as_parquets(tokenized)


def load_tokenized_dataset(data_dir=OUTPUT_DIR):
    """
    TODO: Implement loading of tokenized dataset from parquet files.
    - List only parquet files in data_dir, sorted by filename
    - Load them using load_dataset('parquet', data_files=...)
    - Return the 'train' split
    """

    parquet_files = sorted(
        [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".parquet")
        ]
    )
    dataset = load_dataset("parquet", data_files=parquet_files, split="train")

    return dataset


def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    dataset_size = len(dataset)
    train_dataset = dataset.select(range(validation_size, dataset_size))
    eval_dataset = dataset.select(range(validation_size))

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def create_model(tokenizer):
    # Don't change this parameter
    MODEL_CONFIG = {
        "hidden_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 8192,
        "head_dim": 128,
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "scale_attn_weights": True,
        "use_cache": True,
    }

    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG,
    )

    model = Qwen3ForCausalLM._from_config(
        config, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
    )

    print(f"Model pad token id: {model.config.pad_token_id}")

    with torch.no_grad():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")

    return model


def train_model():
    """
    TODO: Implement the training pipeline.
    - Prepare tokenizer
    - Load tokenized dataset and split it
    - Create the model
    - Create TrainingArguments from TRAINING_CONFIG
    - Create Trainer with TimeoutCallback
    - Train the model
    - Run final evaluation and print results
    - Save metric history to trainer_state.json for local loss plots
    """

    tokenizer = prepare_tokenizer()

    dataset = load_tokenized_dataset()
    train_dataset, eval_dataset = split_dataset(dataset)

    model = create_model(tokenizer)
    training_args = TrainingArguments(**TRAINING_CONFIG)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[
            TimeoutCallback(timeout_seconds=MAX_TRAINING_TIME_SECONDS)
        ],  # dont change
    )
    trainer.train()
    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    trainer.save_state()

    return trainer, tokenizer


def generate_examples(model, tokenizer, prompts, max_new_tokens=100):
    """
    Generate text examples for the report.
    """
    model.eval()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results[prompt] = generated
        print(f"\nPROMPT:\n{prompt}\n")
        print(f"\nGENERATED:\n{generated}\n")

    return results


if __name__ == "__main__":

    print(TRAINING_CONFIG)

    # Step 1: Prepare the dataset (run once)
    prepare_dataset()

    trainer, tokenizer = train_model()

    test_prompts = [
        "Рецепт пирога с яблоками:",
        "У лукоморья дуб зеленый",
        "Столица России - это",
        "Вторая буква в русском алфавите - это",
        "Раз, два, три",
    ]

    print("RESULTS AFTER TRAINING")

    generate_examples(
        trainer.model,
        tokenizer,
        test_prompts,
        max_new_tokens=128,
    )
