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
from protbench.models import initialize_model

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True


@hydra.main(config_name="config", config_path="config", version_base=None)
def main(config_args: omegaconf.DictConfig):
    for env_variable, value in config_args.env_variables.items():
        os.environ[env_variable] = value

    for model_family, checkpoint in zip(config_args.models_family, config_args.model_checkpoints):
        with torch.device("cuda:0"):
            pretrained_model = applications.initialize_model_from_checkpoint(
                model_family,
                checkpoint,
            )
            pretrained_model.initialze_model_from_checkpoint(
                gradient_checkpointing=config_args.train_config.gradient_checkpointing,
            )
            embedding_dim = pretrained_model.embedding_dim

         # Loop over the tasks.
        for task_name, task_cls in applications.load_tasks(
            tasks_to_run=config_args.tasks
        ): 
            # Create a task instance. (e.g. SSP3, SSP8, etc...)
            task = task_cls.initialize(
                dataset=task_name, from_embeddings=True, tokenizer=pretrained_model.tokenizer
            )
            train_seqs, train_labels = task.load_train_data()
            val_seqs, val_labels = task.load_eval_data()

            test_seqs, test_labels = None, None
            if task.test_dataset is not None:
                test_seqs, test_labels = task.load_test_data()

            num_classes = task.num_classes

            # This instance is just a data class where it
            # stores the paths for saving the embeddings.
            save_dirs = embedder.SaveDirectories()

            # Create ComputeEmbeddingsWrapper dataset that loops
            # over the dataset, tokenizes each sequence and
            # extracts the sequence embeddings (last hidden state).
            compute_embeddings_wrapper = embedder.ComputeEmbeddingsWrapper(
                model=pretrained_model.model,
                 # The default tokenization function is just a class that wraps the tokenizer with some default arguments.
                 # You can replace the default tokenization function with any other function you want,
                tokenization_fn=pretrained_model.load_default_tokenization_function(
                    return_input_ids_only=True, tokenizer_options=config_args.tokenizer_config
                ),
                # A simple function that takes the output of the model and modify it if needed
                # or if the model returns an object that has the embedding inside it you will
                # need to pass this function to return the tensor itself.
                post_processing_function=pretrained_model.embeddings_postprocessing_fn,
                pad_token_id=0,
                low_memory=config_args.train_config.low_memory,
                save_directories=save_dirs,
                forward_options={}, # Used in case the forward function requires arguments other than the `input_ids``
            )
            embedding_outputs = compute_embeddings_wrapper(
                train_seqs=train_seqs,
                val_seqs=val_seqs,
                test_seqs=test_seqs,
            )
            # We do not need this model
            # anymore so we free up space.
            pretrained_model.model.cpu()
            torch.cuda.empty_cache()

            # if we cannot store the embeddings in our
            # memory then we save it to the disk.
            if config_args.train_config.low_memory:
                train_dataset = dataset_adapters.EmbeddingsDatasetFromDisk(
                    save_dirs.train, train_labels
                )
                val_dataset = dataset_adapters.EmbeddingsDatasetFromDisk(
                    save_dirs.validation, val_labels
                )
                if task.test_dataset is not None:
                    test_dataset = dataset_adapters.EmbeddingsDatasetFromDisk(
                        save_dirs.test, test_labels
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

                # Load our downstream model.
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

                # Then we connect everything together,
                # by using this function we will create
                # a downstream convbert model with an
                # output layer that corresponds to
                # the current task and we add pooling layer if required.
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
