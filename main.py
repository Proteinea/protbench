import json
import random
from sys import argv

import torch
import numpy as np
from transformers import Trainer

from protbench.src.train_utils import TrainUtils


def seed_everything(seed):
    """Seed everything for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_configs():
    if len(argv) == 1:
        config_path = "config.json"
    else:
        config_path = argv[1]
    config = json.load(open(config_path, "r"))
    return config


def main():
    config = load_configs()
    seed_everything(config["general"]["seed"])

    train_utils = TrainUtils(config=config)
    train_utils.log_info()

    downstream_model = train_utils.downstream_model
    training_args = train_utils.training_args
    train_dataset, val_dataset = train_utils.create_embd_datasets()
    train_utils.delete_pretrained_model()

    trainer = Trainer(
        model=downstream_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=train_utils.get_optimizers(downstream_model),
        compute_metrics=train_utils.compute_metrics,
        data_collator=train_utils.collator_fn,
        preprocess_logits_for_metrics=train_utils.logits_preprocessor_fn,
    )
    trainer.train()


if __name__ == "__main__":
    main()
