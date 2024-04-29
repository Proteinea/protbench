# flake8: noqa: E402

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
from functools import partial

import hydra
import numpy as np
import omegaconf
import torch
import wandb
from transformers import Trainer
from transformers import TrainingArguments

from protbench import applications
from protbench.embedder import utils
from protbench.models import ConvBert
from protbench.utils import EmbeddingsDataset
from protbench.utils import EmbeddingsDatasetFromDisk
from protbench.utils import SequenceAndLabelsDataset


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_name="config", config_path="config", version_base=None)
def main(config_args: omegaconf.DictConfig):
    for env_variable, value in config_args.env_variables.items():
        os.environ[env_variable] = value

    for checkpoint in config_args.model_checkpoints:
        with torch.device("cuda:0"):
            pretrained_model, tokenizer = applications.models.ankh.initialize_model_from_checkpoint(
                checkpoint,
                initialize_with_lora=False,
                gradient_checkpointing=config_args.train_config.gradient_checkpointing,
            )
            embedding_dim = pretrained_model.config.d_model
            tokenizer = partial(
                tokenizer,
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            )

        for task_name, task in applications.get_tasks(
            tasks_to_run=config_args.tasks, from_embeddings=True, tokenizer=tokenizer
        ):
            train_seqs, train_labels = task.get_train_data()
            val_seqs, val_labels = task.get_eval_data()

            if task.test_dataset is not None:
                test_seqs, test_labels = task.get_test_data()
            else:
                test_seqs, test_labels = None, None

            num_classes = task.get_num_classes()

            if task.from_embeddings:
                save_dirs = utils.SaveDirectories()
                compute_embeddings_wrapper = utils.ComputeEmbeddingsWrapper(
                    model=pretrained_model,
                    tokenizer=tokenizer.encode,
                    tokenizer_options={
                        "add_special_tokens": True,
                        "padding": True,
                        "truncation": False,
                        "truncation": False,
                    },
                    post_processing_function=applications.models.ankh.embeddings_postprocessing_fn,
                    pad_token_id=0,
                    low_memory=config_args.train_config.low_memory,
                    save_path=save_dirs,
                )
                embedding_outputs = compute_embeddings_wrapper(train_seqs=train_seqs, val_seqs=val_seqs, test_seqs=test_seqs)
                pretrained_model.cpu() # To free up space.
                torch.cuda.empty_cache()

                if not config_args.train_config.low_memory:
                    train_embeds, val_embeds, test_embeds = embedding_outputs

                    train_dataset = EmbeddingsDataset(train_embeds, train_labels)
                    val_dataset = EmbeddingsDataset(val_embeds, val_labels)
                    if task.test_dataset is not None:
                        test_dataset = EmbeddingsDataset(test_embeds, test_labels)
                else:
                    train_dataset = EmbeddingsDatasetFromDisk(
                        save_dirs.train, train_labels
                    )
                    val_dataset = EmbeddingsDatasetFromDisk(
                        save_dirs.validation, val_labels
                    )
                    if task.test_dataset is not None:
                        test_dataset = EmbeddingsDatasetFromDisk(
                            save_dirs.test, test_labels
                        )
            else:
                train_dataset = SequenceAndLabelsDataset(
                    train_seqs, train_labels
                )
                val_dataset = SequenceAndLabelsDataset(val_seqs, val_labels)

                if task.test_dataset is not None:
                    test_dataset = SequenceAndLabelsDataset(
                        test_seqs, test_labels
                    )

            print("Number of train embeddings: ", len(train_dataset))
            print("Number of validation embeddings: ", len(val_dataset))
            if task.test_dataset is not None:
                print("Number of test embeddings: ", len(test_dataset))
            print("Number of classes: ", num_classes)

            for i in range(config_args.train_config.num_trials_per_checkpoint):
                run_name = f"original-{checkpoint}-{task_name}-{i}-{config_args.convbert_config.pooling}"

                set_seed(config_args.train_config.seed)

                downstream_model = ConvBert(
                    input_dim=embedding_dim,
                    nhead=config_args.convbert_config.nhead,
                    hidden_dim=config_args.convbert_config.hidden_dim
                    or int(embedding_dim / 2),  # noqa
                    num_layers=config_args.convbert_config.num_layers,
                    kernel_size=config_args.convbert_config.kernel_size,
                    dropout=config_args.convbert_config.dropout,
                    pooling=config_args.convbert_config.pooling
                    if task.requires_pooling
                    else None,
                )

                model = task.get_downstream_model(
                    downstream_model,
                    embedding_dim,
                    pooling=config_args.convbert_config.pooling,
                )

                training_args = TrainingArguments(
                    output_dir=os.path.join("trainer-outputs", run_name),
                    run_name=run_name,
                    num_train_epochs=config_args.train_config.num_train_epochs,
                    per_device_train_batch_size=config_args.train_config.per_device_train_batch_size,  # noqa
                    per_device_eval_batch_size=config_args.train_config.per_device_eval_batch_size,  # noqa
                    warmup_steps=config_args.train_config.warmup_steps,
                    learning_rate=config_args.train_config.learning_rate,
                    weight_decay=config_args.train_config.weight_decay,
                    logging_dir=f"./logs_{run_name}",
                    logging_steps=config_args.train_config.logging_steps,
                    do_train=True,
                    do_eval=True,
                    evaluation_strategy=config_args.train_config.evaluation_strategy,  # noqa
                    gradient_accumulation_steps=config_args.train_config.gradient_accumulation_steps,  # noqa
                    fp16=False,
                    fp16_opt_level="02",
                    seed=config_args.train_config.seed,
                    load_best_model_at_end=False,
                    save_total_limit=1,
                    metric_for_best_model=task.metric_for_best_model,
                    greater_is_better=True,
                    save_strategy=config_args.train_config.save_strategy,
                    report_to=config_args.train_config.report_to,
                    remove_unused_columns=False,
                )
                if task.test_dataset is not None:
                    eval_ds = {"validation": val_dataset, "test": test_dataset}
                else:
                    eval_ds = {"validation": val_dataset}

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_ds,
                    compute_metrics=task.metrics_fn,
                    data_collator=task.collate_fn,
                    preprocess_logits_for_metrics=task.preprocessing_fn,
                )
                trainer.train()
                wandb.finish()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("spawn")
    main()
