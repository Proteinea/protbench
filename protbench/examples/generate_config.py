import os
from typing import Dict

import yaml


def generate_config() -> Dict:
    config = {
        "train_config": {
            "num_trials_per_checkpoint": 1,
            "seed": 7,
            "gradient_checkpointing": False,
            "num_train_epochs": 5,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "warmup_steps": 1000,
            "learning_rate": 1e-03,
            "weight_decay": 0.0,
            "logging_steps": 10,
            "evaluation_strategy": "epoch",
            "gradient_accumulation_steps": 16,
            "save_strategy": "epoch",
            "report_to": "wandb",
            "low_memory": True,
        },
        "model_with_lora_config": {
            "pooling": "max",
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_bias": "none",
            "target_modules": ["q", "v"]
        },
        "model_checkpoints": [
            "ankh-base",
            "ankh-v2-23",
            "ankh-v2-32",
            "ankh-v2-33",
            "ankh-v2-41",
            "ankh-v2-45",
            "ankh-large",
            "esm2_650M",
            "esm2_3B",
            "esm2_15B",
        ],
        "tasks": [
            "ssp3_casp12",
            "ssp3_casp14",
            "ssp3_cb513",
            "ssp3_ts115",
            "ssp8_casp14",
            "ssp8_casp12",
            "ssp8_cb513",
            "ssp8_ts115",
            "deeploc",
            "solubility",
            "remote_homology",
            "fluorescence",
            "gb1_sampled",
            "thermostability",
        ],
        "convbert_config": {
            "nhead": 4,
            "hidden_dim": 768,
            "num_layers": 1,
            "kernel_size": 7,
            "dropout": 0.1,
            "pooling": "avg",
        },
        "env_variables": {
            "WANDB_PROJECT": "Benchmarking",
        },
        "tokenizer_config": {
            "max_length": None,
            "padding": False,
            "truncation": False,
        },
    }
    return config


def save_yaml(config: Dict):
    if not os.path.exists("config"):
        os.mkdir("config")

    with open("config/config.yaml", "w") as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    config = generate_config()
    save_yaml(config)
    print("Generated Config and Saved.")
