# flake8: noqa: E402

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import gc

import hydra
import omegaconf
import torch
import wandb
from transformers import Trainer
from transformers import TrainingArguments

from protbench import applications
from protbench.examples.utils import create_run_name
from protbench.examples.utils import set_seed
from protbench.utils import SequenceAndLabelsDataset
from protbench.models import initialize_model


@hydra.main(config_name="config", config_path="config", version_base=None)
def main(config_args: omegaconf.DictConfig):
    for env_variable, value in config_args.env_variables.items():
        os.environ[env_variable] = value

    for model_family, checkpoint in zip(config_args.models_family, config_args.model_checkpoints):
        for task_name, task_cls in applications.load_tasks(
            tasks_to_run=config_args.tasks
        ):
            with torch.device("cuda:0"):
                pretrained_model = applications.initialize_model_from_checkpoint(
                    model_family,
                    checkpoint,
                )

                pretrained_model.initialze_model_from_checkpoint_with_lora(
                    lora_r=config_args.model_with_lora_config.lora_r,
                    lora_alpha=config_args.model_with_lora_config.lora_alpha,
                    lora_dropout=config_args.model_with_lora_config.lora_dropout,  # noqa
                    lora_bias=config_args.model_with_lora_config.lora_bias,
                    target_modules=config_args.model_with_lora_config.target_modules,
                    gradient_checkpointing=config_args.train_config.gradient_checkpointing,
                )
                embedding_dim = pretrained_model.embedding_dim

                # You can still use the loaded tokenizer 
                # instead of calling this function by directly
                # accessing the tokenizer using `pretrained_model.tokenzier`.
                # This class that will be returned from this function
                # just does some extra pre and post processing depending on the model
                # so its convenient to use it instead.
                tokenizer = pretrained_model.load_default_tokenization_function(
                    return_input_ids_only=False,
                    tokenizer_options={
                        "add_special_tokens": True,
                        "padding": config_args.tokenizer_config.padding,
                        "max_length": config_args.tokenizer_config.max_length,
                        "truncation": config_args.tokenizer_config.truncation,
                        "return_tensors": "pt"},
                )

            task = task_cls.initialize(
                dataset=task_name, from_embeddings=False, tokenizer=tokenizer
            )

            train_seqs, train_labels = task.load_train_data()
            val_seqs, val_labels = task.load_eval_data()
            if task.test_dataset is not None:
                test_seqs, test_labels = task.load_test_data()
            num_classes = task.num_classes

            train_dataset = SequenceAndLabelsDataset(train_seqs, train_labels)
            val_dataset = SequenceAndLabelsDataset(val_seqs, val_labels)
            if task.test_dataset is not None:
                test_dataset = SequenceAndLabelsDataset(test_seqs, test_labels)

            print("Number of train embeddings: ", len(train_dataset))
            print("Number of validation embeddings: ", len(val_dataset))
            if task.test_dataset is not None:
                print("Number of test embeddings: ", len(test_dataset))
            print("Number of classes: ", num_classes)

            for i in range(config_args.train_config.num_trials_per_checkpoint):
                run_name = create_run_name(
                    num_trial=i,
                    checkpoint=checkpoint,
                    task_name=task_name,
                    pooling=config_args.model_with_lora_config.pooling,
                    lora_r=config_args.model_with_lora_config.lora_r,
                    lora_alpha=config_args.model_with_lora_config.lora_alpha,
                    target_modules=list(config_args.model_with_lora_config.target_modules),
                )
                set_seed(config_args.train_config.seed)

                model = initialize_model(
                    task=task,
                    embedding_dim=embedding_dim,
                    from_embeddings=False,
                    backbone=pretrained_model.model,
                    downstream_model=None,
                    pooling=config_args.model_with_lora_config.pooling
                    if task.requires_pooling
                    else None,
                    embedding_postprocessing_fn=pretrained_model.embeddings_postprocessing_fn,
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
                    load_best_model_at_end=True,
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
                del pretrained_model
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("spawn")
    main()
