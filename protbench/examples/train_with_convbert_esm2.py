# flake8: noqa: E402

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import hydra
import omegaconf
import torch
import wandb
from transformers import Trainer
from transformers import TrainingArguments

from protbench import applications
from protbench import embedder
from protbench.examples.utils import create_run_name
from protbench.examples.utils import set_seed
from protbench.models import ConvBert
from protbench.utils import dataset_adapters
from protbench.models.utils import initialize_model


@hydra.main(config_name="config", config_path="config", version_base=None)
def main(config_args: omegaconf.DictConfig):
    for env_variable, value in config_args.env_variables.items():
        os.environ[env_variable] = value

    for checkpoint in config_args.model_checkpoints:
        with torch.device("cuda:0"):
            (
                pretrained_model,
                tokenizer,
            ) = applications.pretrained.esm2.initialize_model_from_checkpoint(
                checkpoint,
                initialize_with_lora=False,
                gradient_checkpointing=config_args.train_config.gradient_checkpointing,
            )
            embedding_dim = pretrained_model.embed_dim

        for task_name, task_cls in applications.get_tasks(
            tasks_to_run=config_args.tasks
        ):
            task = task_cls(
                dataset=task_name, from_embeddings=True, tokenizer=tokenizer
            )
            train_seqs, train_labels = task.get_train_data()
            val_seqs, val_labels = task.get_eval_data()

            if task.test_dataset is not None:
                test_seqs, test_labels = task.get_test_data()
            else:
                test_seqs, test_labels = None, None

            num_classes = task.get_num_classes()

            save_dirs = embedder.SaveDirectories()
            compute_embeddings_wrapper = embedder.ComputeEmbeddingsWrapper(
                model=pretrained_model,
                 # The default tokenization function is just a class that wraps the tokenizer with some default arguments.
                 # You can replace the default tokenization function with any other function you want,
                tokenization_fn=applications.pretrained.esm2.DefaultTokenizationFunction(tokenizer),
                # A simple function that takes the output of the model and modify it if needed
                # or if the model returns an object that has the embedding inside it you will
                # need to pass this function to return the tensor itself.
                post_processing_function=applications.pretrained.esm2.embeddings_postprocessing_fn,
                pad_token_id=0,
                low_memory=config_args.train_config.low_memory,
                save_directories=save_dirs,
                forward_options=None,
            )
            embedding_outputs = compute_embeddings_wrapper(
                train_seqs=train_seqs,
                val_seqs=val_seqs,
                test_seqs=test_seqs,
            )
            # We do not need this model
            # anymore so we free up space.
            pretrained_model.cpu()
            torch.cuda.empty_cache()

            if config_args.train_config.low_memory:
                train_dataset = dataset_adapters.EmbeddingsDatasetFromDisk(
                    save_dirs.train_path, train_labels
                )
                val_dataset = dataset_adapters.EmbeddingsDatasetFromDisk(
                    save_dirs.validation_path, val_labels
                )
                if task.test_dataset is not None:
                    test_dataset = dataset_adapters.EmbeddingsDatasetFromDisk(
                        save_dirs.test_path, test_labels
                    )
            else:
                train_embeds, val_embeds, test_embeds = embedding_outputs
                train_dataset = dataset_adapters.EmbeddingsDataset(
                    train_embeds, train_labels
                )
                val_dataset = dataset_adapters.EmbeddingsDataset(
                    val_embeds, val_labels
                )
                if task.test_dataset is not None:
                    test_dataset = dataset_adapters.EmbeddingsDataset(
                        test_embeds, test_labels
                    )

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
                    pooling=config_args.convbert_config.pooling,
                )

                set_seed(config_args.train_config.seed)

                downstream_model = ConvBert(
                    input_dim=embedding_dim,
                    nhead=config_args.convbert_config.nhead,
                    hidden_dim=config_args.convbert_config.hidden_dim
                    or int(embedding_dim / 2),
                    num_layers=config_args.convbert_config.num_layers,
                    kernel_size=config_args.convbert_config.kernel_size,
                    dropout=config_args.convbert_config.dropout,
                    pooling=config_args.convbert_config.pooling
                    if task.requires_pooling
                    else None,
                )

                model = initialize_model(
                    task=task,
                    embedding_dim=embedding_dim,
                    from_embeddings=True,
                    # No need for the backbone because
                    # we already used it to extract embeddings,
                    # now we will just use the extracted embeddings
                    # to pass it to the downstream model which
                    # we passed to this function..
                    backbone=None,
                    downstream_model=downstream_model,
                    pooling=config_args.convbert_config.pooling,
                    # No need to pass embedding postprocessing
                    # function because we do not have a
                    # pretrained backbone.
                    embedding_postprocessing_fn=None,
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
