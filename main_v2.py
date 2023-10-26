import os

import wandb
from functools import partial

from protbench.embedder import TorchEmbedder, TorchEmbeddingFunction
from protbench.tasks import (
    PickleResidueToClass
)
from protbench.models import (
    ConvBert,
    DownstreamModelFromEmbedding,
    ContactPredictionHead,
)
from protbench import metrics
from protbench.utils import (
    collate_inputs_and_labels,
)
from protbench import applications
import torch
import numpy as np
import random
from transformers import (
    Trainer,
    TrainingArguments,
)
from peft import TaskType
from protbench.utils import SequenceAndLabelsDataset, EmbeddingsDataset, EmbeddingsDatasetFromDisk # noqa

from scipy.spatial.distance import pdist, squareform

os.environ["WANDB_PROJECT"] = "AnkhV2-LoRA"


def preprocess_contact_prediction_labels(seq, label, mask):
    contact_map = np.less(squareform(pdist(label)), 8.0).astype(np.int64)
    yind, xind = np.indices(contact_map.shape)
    invalid_mask = ~(mask[:, None] & mask[None, :])
    invalid_mask |= np.abs(yind - xind) < 6
    contact_map[invalid_mask] = -1
    return seq, contact_map, mask


def get_data(task_name, max_seqs=None):
    if task_name == "contact_prediction":
        train_data = PickleResidueToClass(
            **{
                "dataset_path": "contact_prediction/train.pickle",
                "seqs_col": "primary",
                "labels_col": "tertiary",
                "mask_col": "valid_mask",
                "preprocessing_function": preprocess_contact_prediction_labels,
                "num_classes": 2,
            }
        )
        val_data = PickleResidueToClass(
            **{
                "dataset_path": "contact_prediction/valid.pickle",
                "seqs_col": "primary",
                "labels_col": "tertiary",
                "mask_col": "valid_mask",
                "preprocessing_function": preprocess_contact_prediction_labels,
                "num_classes": 2,
            },
        )

    return (
        train_data.data[0][:max_seqs],
        train_data.data[1][:max_seqs],
        val_data.data[0][:max_seqs],
        val_data.data[1][:max_seqs],
        getattr(train_data, "num_classes", None),
    )


def embeddings_postprocessing_fn(model_outputs):
    return model_outputs.last_hidden_state


def tokenize(batch, tokenizer):
    return tokenizer.batch_encode_plus(
        batch,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    )["input_ids"]


def delete_directory_contents(directory_path):
    for root, _, files in os.walk(directory_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))


def compute_embeddings(model, tokenizer, train_seqs, val_seqs):
    embedding_fn = TorchEmbeddingFunction(
        model,
        partial(tokenize, tokenizer=tokenizer),
        device="cuda:0",
        embeddings_postprocessing_fn=embeddings_postprocessing_fn,
        pad_token_id=tokenizer.pad_token_id,
    )
    embedder = TorchEmbedder(
        embedding_fn,
        low_memory=False,
        save_path=None,
        devices=None,
        batch_size=1,
    )

    embeddings = []
    for data in [train_seqs, val_seqs]:
        embeddings.append(embedder.run(data))

    return embeddings


def compute_embeddings_and_save_to_disk(model, tokenizer, train_seqs, val_seqs):
    embedding_fn = TorchEmbeddingFunction(
        model,
        partial(tokenize, tokenizer=tokenizer),
        device="cuda:0",
        embeddings_postprocessing_fn=embeddings_postprocessing_fn,
        pad_token_id=tokenizer.pad_token_id,
    )

    train_embeddings_path = 'train_embeddings'
    val_embeddings_path = 'val_embeddings'

    if not os.path.exists(train_embeddings_path):
        os.mkdir(train_embeddings_path)
    else:
        delete_directory_contents(train_embeddings_path)

    if not os.path.exists(val_embeddings_path):
        os.mkdir(val_embeddings_path)
    else:
        delete_directory_contents(val_embeddings_path)

    for data, path in [(train_seqs, train_embeddings_path), (val_seqs, val_embeddings_path)]:
        embedder = TorchEmbedder(
            embedding_fn,
            low_memory=True,
            save_path=path,
            devices=None,
            batch_size=1,
        )
        embedder.run(data)


def get_downstream_model(task_name, embedding_dim, num_classes):
    convbert_args = {
        "input_dim": embedding_dim,
        "nhead": 4,
        "hidden_dim": int(embedding_dim / 2),
        "num_layers": 1,
        "kernel_size": 7,
        "dropout": 0.2,
    }

    task_class_map = {
        "contact_prediction": (
            ConvBert,
            {
                "pooling": None,
                **convbert_args,
            },
            ContactPredictionHead,
            {
                "input_dim": embedding_dim,
                "output_dim": num_classes,
            },
        ),
    }
    return DownstreamModelFromEmbedding(
        task_class_map[task_name][0](**task_class_map[task_name][1]),
        task_class_map[task_name][2](**task_class_map[task_name][3]),
    )


def get_metrics(task_name, num_classes=None):
    task_class_map = {
        "contact_prediction": None,
    }
    return task_class_map[task_name]

def get_collate_fn(task_name):
    task_class_map = {
        "contact_prediction": collate_inputs_and_labels,
    }
    return task_class_map[task_name]


def get_logits_preprocessing_fn(task_name):
    task_class_map = {
        "contact_prediction": None,
    }
    return task_class_map[task_name]


def get_metric_for_best_model(task_name):
    task_metric_map = {
        "contact_prediction": "eval_loss",
    }
    return task_metric_map[task_name]


def compute_error_bar(task_name):
    task_map = {
        "ssp-casp12": metrics.compute_error_bar_for_token_classification,
        "ssp-casp14": metrics.compute_error_bar_for_token_classification,
        "ssp-cb513": metrics.compute_error_bar_for_token_classification,
        "ssp-ts115": metrics.compute_error_bar_for_token_classification,
        "ssp8-casp12": metrics.compute_error_bar_for_token_classification,
        "ssp8-casp14": metrics.compute_error_bar_for_token_classification,
        "ssp8-cb513": metrics.compute_error_bar_for_token_classification,
        "ssp8-ts115": metrics.compute_error_bar_for_token_classification,
        "solubility": metrics.compute_error_bar_for_binary_classification,
        "fluorescence": metrics.compute_error_bar_for_regression,
        # "contact_prediction": metrics.compute_error_bar_for_token_classification
        "deeploc": metrics.compute_error_bar_for_token_classification,
        "remote_homology": metrics.compute_error_bar_for_token_classification,
    }
    return task_map[task_name]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def available_tasks(pooling='max'):
    tasks = {
        'ssp3_casp12': partial(applications.SSP3, dataset='ssp3_casp12', from_embeddings=False, task_type=TaskType.TOKEN_CLS),
        'ssp3_casp14': partial(applications.SSP3, dataset='ssp3_casp14', from_embeddings=False, task_type=TaskType.TOKEN_CLS),
        'ssp3_cb513': partial(applications.SSP3, dataset='ssp3_cb513', from_embeddings=False, task_type=TaskType.TOKEN_CLS),
        'ssp3_ts115': partial(applications.SSP3, dataset='ssp3_ts115', from_embeddings=False, task_type=TaskType.TOKEN_CLS),
        "ssp8_casp12": partial(applications.SSP8, dataset="ssp8_casp12", from_embeddings=False, task_type=TaskType.TOKEN_CLS),
        "ssp8_casp14": partial(applications.SSP8, dataset="ssp8_casp14", from_embeddings=False, task_type=TaskType.TOKEN_CLS),
        "ssp8_cb513": partial(applications.SSP8, dataset="ssp8_cb513", from_embeddings=False, task_type=TaskType.TOKEN_CLS),
        "ssp8_ts115": partial(applications.SSP8, dataset="ssp8_ts115", from_embeddings=False, task_type=TaskType.TOKEN_CLS),
        "deeploc": partial(applications.DeepLoc, dataset="deeploc", from_embeddings=False, task_type=TaskType.SEQ_CLS),
        "solubility": partial(applications.Solubility, from_embeddings=False, task_type=TaskType.SEQ_CLS),
        "remote_homology": partial(applications.RemoteHomology, from_embeddings=False, task_type=TaskType.SEQ_CLS),
        "fluorescence": partial(applications.Fluorescence, from_embeddings=False, task_type=TaskType.SEQ_CLS),
    }
    for task_name, task in tasks.items():
        yield task_name, task



def main():
    NUM_TRIALS_PER_CHECKPOINT = 1
    SEED = 7
    MAX_SEQS = None
    LOW_MEMORY = True
    POOLING = 'max'
    USE_LORA = True
    GRADIENT_CHECKPOINTING = True

    checkpoints = [
        "ankh-base",
        "ankh-v2-23",
        "ankh-v2-32",
        "ankh-v2-33",
        "ankh-v2-41",
        "ankh-v2-45",
        "ankh-large",
    ]

    for checkpoint in checkpoints:
        for task_name, task in available_tasks():
            with torch.device('cuda:0'):
                pretrained_model, tokenizer = applications.models.ankh.initialize_model_from_checkpoint(
                    checkpoint, initialize_with_lora=USE_LORA, task_type=task.task_type,
                )
                if GRADIENT_CHECKPOINTING:
                    pretrained_model.gradient_checkpointing_enable()
                embedding_dim = pretrained_model.config.d_model
                tokenizer = partial(tokenizer, add_special_tokens=True,
                                    padding='longest',
                                    return_tensors="pt")

            task = task(tokenizer=tokenizer)
            task: applications.BenchmarkingTask
            train_seqs, train_labels = task.get_train_data()
            val_seqs, val_labels = task.get_eval_data()
            num_classes = task.get_num_classes()
            if not LOW_MEMORY and task.from_embeddings:
                train_embds, val_embds = compute_embeddings(
                    pretrained_model, tokenizer, train_seqs, val_seqs
                )
            elif task.from_embeddings:
                compute_embeddings_and_save_to_disk(
                    pretrained_model, tokenizer, train_seqs, val_seqs
                )
            if task.from_embeddings:
                pretrained_model.cpu()
                torch.cuda.empty_cache()

            collate_fn = task.collate_fn
            logits_preprocessing_fn = task.preprocessing_fn

            if not LOW_MEMORY and task.from_embeddings:
                train_dataset = EmbeddingsDataset(train_embds, train_labels)
                val_dataset = EmbeddingsDataset(val_embds, val_labels)
            elif task.from_embeddings:
                train_dataset = EmbeddingsDatasetFromDisk('train_embeddings',
                                                          train_labels)
                val_dataset = EmbeddingsDatasetFromDisk('val_embeddings',
                                                        val_labels)
            else:
                train_dataset = SequenceAndLabelsDataset(train_seqs, train_labels)
                val_dataset = SequenceAndLabelsDataset(val_seqs, val_labels)

            print("Number of train embeddings: ", len(train_dataset))
            print("Number of validation embeddings: ", len(val_dataset))
            print("Number of classes: ", num_classes)

            for i in range(NUM_TRIALS_PER_CHECKPOINT):
                run_name = f"original-{checkpoint}-{task_name}-{i}"
                set_seed(SEED)
                model = task.get_downstream_model(
                    pretrained_model, embedding_dim, pooling=POOLING if task.requires_pooling else None
                )

                training_args = TrainingArguments(
                    output_dir=os.path.join("trainer-outputs", run_name),
                    run_name=run_name,
                    num_train_epochs=5,
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    warmup_steps=1000,
                    learning_rate=1e-03,
                    weight_decay=0.0,
                    logging_dir=f"./logs_{run_name}",
                    logging_steps=10,
                    do_train=True,
                    do_eval=True,
                    evaluation_strategy="epoch",
                    gradient_accumulation_steps=16,
                    fp16=False,
                    fp16_opt_level="02",
                    seed=SEED,
                    load_best_model_at_end=True,
                    save_total_limit=1,
                    metric_for_best_model=task.metric_for_best_model,
                    greater_is_better=True,
                    save_strategy="epoch",
                    report_to="wandb",
                    remove_unused_columns=False,
                )
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    compute_metrics=task.metrics_fn,
                    data_collator=collate_fn,
                    preprocess_logits_for_metrics=logits_preprocessing_fn,
                    # callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
                )
                trainer.train()

                wandb.finish()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("spawn")
    main()
