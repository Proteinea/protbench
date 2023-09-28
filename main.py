import os

os.environ["WANDB_PROJECT"] = "AnkhV2"

import wandb
from functools import partial

from protbench.embedder import TorchEmbedder, TorchEmbeddingFunction
from protbench.tasks import (
    HuggingFaceResidueToClass,
    HuggingFaceSequenceToClass,
    HuggingFaceSequenceToValue,
    PickleResidueToClass
)
from protbench.models import (
    ConvBert,
    TokenClassificationHead,
    BinaryClassificationHead,
    RegressionHead,
    DownstreamModel,
    ContactPredictionHead,
    MultiClassClassificationHead
)
from protbench import metrics
from protbench.utils import (
    collate_inputs,
    collate_inputs_and_labels,
    preprocess_multi_classification_logits,
    preprocess_binary_classification_logits,
)

from torch.utils.data import Dataset
import torch
import numpy as np
import random
from transformers import (
    T5EncoderModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, top_k_accuracy_score
import glob


class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, labels, shift_left=0, shift_right=1):
        """Dataset for embeddings and corresponding labels of a task.

        Args:
            embeddings (list[torch.Tensor]): list of tensors of embeddings (batch_size, seq_len, embd_dim)
                where each tensor may have a different seq_len.
            labels (list[Any]): list of labels.
        """
        if len(embeddings) != len(labels):
            raise ValueError(
                "embeddings and labels must have the same length but got "
                f"{len(embeddings)} and {len(labels)}"
            )
        self.embeddings = embeddings
        self.labels = labels
        self.shift_left = shift_left
        self.shift_right = shift_right

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embds = self.embeddings[idx][self.shift_left : -self.shift_right, :]
        labels = torch.tensor(self.labels[idx])
        return {
            "embds": embds,
            "labels": labels,
        }


class EmbeddingsDatasetFromDisk(Dataset):
    def __init__(self, embeddings_path, labels, shift_left=0, shift_right=1):
        """Dataset for embeddings and corresponding labels of a task.

        Args:
            embeddings (list[torch.Tensor]): list of tensors of embeddings (batch_size, seq_len, embd_dim)
                where each tensor may have a different seq_len.
            labels (list[Any]): list of labels.
        """
        embeddings_path = sorted(glob.glob(os.path.join(embeddings_path, '*.npy')), key=lambda x: x.split('/')[-1].split('.')[0])
        if len(embeddings_path) != len(labels):
            raise ValueError(
                "embeddings and labels must have the same length but got "
                f"{len(embeddings_path)} and {len(labels)}"
            )
        self.embeddings = embeddings_path
        self.labels = labels
        self.shift_left = shift_left
        self.shift_right = shift_right

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embds = torch.from_numpy(np.load(self.embeddings[idx]))[self.shift_left : -self.shift_right, :]
        labels = torch.tensor(self.labels[idx])
        return {
            "embds": embds,
            "labels": labels,
        }


def preprocess_ssp_rows(seq, label, mask):
    mask = list(map(float, mask.split()))
    return seq, label, mask


def preprocess_contact_prediction_labels(seq, label, mask):
    contact_map = np.less(squareform(pdist(label)), 8.0).astype(np.int64)
    yind, xind = np.indices(contact_map.shape)
    invalid_mask = ~(mask[:, None] & mask[None, :])
    invalid_mask |= np.abs(yind - xind) < 6
    contact_map[invalid_mask] = -1
    return seq, contact_map, mask


def compute_deep_localization_metrics(p):
    prfs = precision_recall_fscore_support(p.label_ids, p.predictions.argmax(axis=1), average='macro')
    return {
        "accuracy": accuracy_score(p.label_ids, p.predictions.argmax(axis=1)),
        "precision": prfs[0],
        "recall": prfs[1],
        "f1": prfs[2],
    }


def compute_remote_homology_metrics(p, num_classes):
    prfs = precision_recall_fscore_support(p.label_ids, p.predictions.argmax(axis=1), average='macro')

    return {
        "accuracy": accuracy_score(p.label_ids, p.predictions.argmax(axis=1)),
        "precision": prfs[0],
        "recall": prfs[1],
        "f1": prfs[2],
        "hits10": top_k_accuracy_score(p.label_ids, p.predictions, k=10,
                                       labels=np.arange(num_classes)),
    }



def get_data(task_name, max_seqs=None):
    if task_name == "ssp-casp12":
        train_data = HuggingFaceResidueToClass(
            **{
                "dataset_url": "proteinea/secondary_structure_prediction",
                "data_files": "training_hhblits.csv",
                "data_key": "train",
                "seqs_col": "input",
                "labels_col": "dssp3",
                "mask_col": "disorder",
                "preprocessing_function": preprocess_ssp_rows,
            }
        )
        val_data = HuggingFaceResidueToClass(
            class_to_id=train_data.class_to_id,
            **{
                "dataset_url": "proteinea/secondary_structure_prediction",
                "data_files": "CASP12.csv",
                "data_key": "train",
                "seqs_col": "input",
                "labels_col": "dssp3",
                "mask_col": "disorder",
                "preprocessing_function": preprocess_ssp_rows,
            },
        )
    elif task_name == "ssp-casp14":
        train_data = HuggingFaceResidueToClass(
            **{
                "dataset_url": "proteinea/secondary_structure_prediction",
                "data_files": "training_hhblits.csv",
                "data_key": "train",
                "seqs_col": "input",
                "labels_col": "dssp3",
                "mask_col": "disorder",
                "preprocessing_function": preprocess_ssp_rows,
            }
        )
        val_data = HuggingFaceResidueToClass(
            class_to_id=train_data.class_to_id,
            **{
                "dataset_url": "proteinea/secondary_structure_prediction",
                "data_files": "CASP14.csv",
                "data_key": "train",
                "seqs_col": "input",
                "labels_col": "dssp3",
                "mask_col": "disorder",
                "preprocessing_function": preprocess_ssp_rows,
            },
        )
    elif task_name == "solubility":
        train_data = HuggingFaceSequenceToClass(
            **{
                "dataset_url": "proteinea/Solubility",
                "data_files": None,
                "data_key": "train",
                "seqs_col": "sequences",
                "labels_col": "labels",
            }
        )
        val_data = HuggingFaceSequenceToClass(
            class_to_id=train_data.class_to_id,
            **{
                "dataset_url": "proteinea/Solubility",
                "data_files": None,
                "data_key": "validation",
                "seqs_col": "sequences",
                "labels_col": "labels",
            },
        )
    elif task_name == "fluorescence":
        train_data = HuggingFaceSequenceToValue(
            **{
                "dataset_url": "proteinea/Fluorosence",
                "data_files": None,
                "data_key": "train",
                "seqs_col": "primary",
                "labels_col": "log_fluorescence",
            }
        )
        val_data = HuggingFaceSequenceToValue(
            **{
                "dataset_url": "proteinea/Fluorosence",
                "data_files": None,
                "data_key": "validation",
                "seqs_col": "primary",
                "labels_col": "log_fluorescence",
            }
        )
    elif task_name == "contact_prediction":
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

    elif task_name == 'deeploc':
        train_data = HuggingFaceSequenceToClass(
            **{
                "dataset_url": "proteinea/deeploc",
                "seqs_col": "input",
                "labels_col": "loc",
                "data_files": None,
                "data_key": "train",
            }
        )

        val_data = HuggingFaceSequenceToClass(
            class_to_id=train_data.class_to_id,
            **{
                "dataset_url": "proteinea/deeploc",
                "seqs_col": "input",
                "labels_col": "loc",
                "data_files": None,
                "data_key": "test",
            }
        )

    elif task_name == 'remote_homology':
        train_data = HuggingFaceSequenceToClass(
            **{
                "dataset_url": "proteinea/remote_homology",
                "seqs_col": "primary",
                "labels_col": "fold_label",
                "data_files": None,
                "data_key": "train",
            }
        )

        val_data = HuggingFaceSequenceToClass(
            class_to_id=train_data.class_to_id,
            **{
                "dataset_url": "proteinea/remote_homology",
                "seqs_col": "primary",
                "labels_col": "fold_label",
                "data_files": None,
                "data_key": "test",
            }
        )

    return (
        train_data.data[0][:max_seqs],
        train_data.data[1][:max_seqs],
        val_data.data[0][:max_seqs],
        val_data.data[1][:max_seqs],
        getattr(train_data, "num_classes", None),
    )


def get_pretrained_model_and_tokenizer(model_name):
    model_url_map = {
        "ankh-base": "ElnaggarLab/ankh-base",
        "ankh-large": "ElnaggarLab/ankh-large",
        "ankh-v2-23": "proteinea-ea/ankh-v2-large-23epochs-a3ee1d6115a726fe83f96d96f76489ff2788143c",
        "ankh-v2-32": "proteinea-ea/ankh-v2-large-32epochs-f60c3a7c8e07fe26bdba04670ab1997f4b679969",
        "ankh-v2-33": "proteinea-ea/ankh-v2-large-33epochs-218254e2e0546838d1427f7f6c32c0cb4664da72",
        "ankh-v2-41": "proteinea-ea/ankh-v2-large-41epochs-e4a2c3615ff005e5e7b5bbd33ec0654106b64f1a",
        "ankh-v2-45": "proteinea-ea/ankh-v2-large-45epochs-62fe367d20d957efdf6e8afe6ae1c724f5bc6775",
    }
    return T5EncoderModel.from_pretrained(
        model_url_map[model_name]
    ), AutoTokenizer.from_pretrained(model_url_map[model_name])


def embeddings_postprocessing_fn(model_outputs):
    return model_outputs[0]


def tokenize(batch, tokenizer):
    return tokenizer.batch_encode_plus(
        batch,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    )["input_ids"]


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

    for data, path in [(train_seqs, 'train_embeddings'), (val_seqs, 'val_embeddings')]:
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
        "ssp-casp12": (
            ConvBert,
            {
                "pooling": None,
                **convbert_args,
            },
            TokenClassificationHead,
            {
                "input_dim": embedding_dim,
                "output_dim": num_classes,
            },
        ),
        "ssp-casp14": (
            ConvBert,
            {
                "pooling": None,
                **convbert_args,
            },
            TokenClassificationHead,
            {
                "input_dim": embedding_dim,
                "output_dim": num_classes,
            },
        ),
        "solubility": (
            ConvBert,
            {
                "pooling": "max",
                **convbert_args,
            },
            BinaryClassificationHead,
            {
                "input_dim": embedding_dim,
            },
        ),
        "fluorescence": (
            ConvBert,
            {
                "pooling": "max",
                **convbert_args,
            },
            RegressionHead,
            {
                "input_dim": embedding_dim,
            },
        ),
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

        "deeploc": (
            ConvBert,
            {
                "pooling": "max",
                **convbert_args,
            },
            MultiClassClassificationHead,
            {
                "input_dim": embedding_dim,
                "output_dim": num_classes,
            }
        ),
        "remote_homology": (
            ConvBert,
            {
                "pooling": "max",
                **convbert_args,
            },
            MultiClassClassificationHead,
            {
                "input_dim": embedding_dim,
                "output_dim": num_classes,
            }
        ),

    }
    return DownstreamModel(
        task_class_map[task_name][0](**task_class_map[task_name][1]),
        task_class_map[task_name][2](**task_class_map[task_name][3]),
    )


def get_metrics(task_name, num_classes=None):
    task_class_map = {
        "ssp-casp12": lambda x: {
            "accuracy": metrics.compute_accuracy(x),
            "precision": metrics.compute_precision(x, average="macro"),
            "recall": metrics.compute_recall(x, average="macro"),
            "f1": metrics.compute_f1(x, average="macro"),
        },
        "ssp-casp14": lambda x: {
            "accuracy": metrics.compute_accuracy(x),
            "precision": metrics.compute_precision(x, average="macro"),
            "recall": metrics.compute_recall(x, average="macro"),
            "f1": metrics.compute_f1(x, average="macro"),
        },
        "solubility": lambda x: {
            "accuracy": metrics.compute_accuracy(x),
            "precision": metrics.compute_precision(x, average="binary"),
            "recall": metrics.compute_recall(x, average="binary"),
            "f1": metrics.compute_f1(x, average="binary"),
        },
        "fluorescence": lambda x: {
            "spearman": metrics.compute_spearman(x),
        },
        "contact_prediction": None,
        "deeploc": lambda x: compute_deep_localization_metrics(x),
        "remote_homology": lambda x: compute_remote_homology_metrics(x, num_classes)
    }
    return task_class_map[task_name]

def get_collate_fn(task_name):
    task_class_map = {
        "ssp-casp12": collate_inputs_and_labels,
        "ssp-casp14": collate_inputs_and_labels,
        "solubility": collate_inputs,
        "fluorescence": collate_inputs,
        "contact_prediction": collate_inputs_and_labels,
        "deeploc": collate_inputs,
        "remote_homology": collate_inputs,
    }
    return task_class_map[task_name]


def get_logits_preprocessing_fn(task_name):
    task_class_map = {
        "ssp-casp12": preprocess_multi_classification_logits,
        "ssp-casp14": preprocess_multi_classification_logits,
        "solubility": preprocess_binary_classification_logits,
        "fluorescence": None,
        "contact_prediction": None,
        "deeploc": None,
        "remote_homology": None,
    }
    return task_class_map[task_name]


def get_metric_for_best_model(task_name):
    task_metric_map = {
        "ssp-casp12": "accuracy",
        "ssp-casp14": "accuracy",
        "solubility": "accuracy",
        "fluorescence": "spearman",
        "contact_prediction": "eval_loss",
        "deeploc": "eval_accuracy",
        "remote_homology": "eval_hits10",
    }
    return task_metric_map[task_name]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    NUM_TRIALS_PER_CHECKPOINT = 5
    SEED = 7
    MAX_SEQS = None
    LOW_MEMORY = True

    checkpoints = [
        # "ankh-base",
        # "ankh-large",
        "ankh-v2-23",
        "ankh-v2-32",
        "ankh-v2-33",
        "ankh-v2-41",
        "ankh-v2-45",
    ]
    tasks = [
        "remote_homology",
        "deeploc",
        # "contact_prediction",
        "ssp-casp14",
        "ssp-casp12",
        "solubility",
        "fluorescence",
    ]

    for checkpoint in checkpoints:
        with torch.device('cuda:0'):
            pretrained_model, tokenizer = get_pretrained_model_and_tokenizer(
                checkpoint
            )
        for task in tasks:
            (
                train_seqs,
                train_labels,
                val_seqs,
                val_labels,
                num_classes,
            ) = get_data(task, max_seqs=MAX_SEQS)
            if not LOW_MEMORY:
                train_embds, val_embds = compute_embeddings(
                    pretrained_model, tokenizer, train_seqs, val_seqs
                )
            else:
                compute_embeddings_and_save_to_disk(
                    pretrained_model, tokenizer, train_seqs, val_seqs
                )
            pretrained_model.cpu()
            torch.cuda.empty_cache()
            collate_fn = get_collate_fn(task)
            logits_preprocessing_fn = get_logits_preprocessing_fn(task)

            if not LOW_MEMORY:
                train_dataset = EmbeddingsDataset(train_embds, train_labels)
                val_dataset = EmbeddingsDataset(val_embds, val_labels)
            else:
                train_dataset = EmbeddingsDatasetFromDisk('train_embeddings',
                                                          train_labels)
                val_dataset = EmbeddingsDatasetFromDisk('val_embeddings',
                                                        val_labels)

            print("Number of train embeddings: ", len(train_dataset))
            print("Number of validation embeddings: ", len(val_dataset))
            print("Number of classes: ", num_classes)

            for i in range(NUM_TRIALS_PER_CHECKPOINT):
                run_name = f"original-{checkpoint}-{task}-{i}"
                set_seed(SEED)
                model = get_downstream_model(
                    task,
                    train_dataset[0]['embds'].shape[1],
                    num_classes,
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
                    metric_for_best_model=get_metric_for_best_model(task),
                    greater_is_better=True,
                    save_strategy="epoch",
                    report_to="wandb",
                )
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    compute_metrics=get_metrics(task, num_classes),
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
