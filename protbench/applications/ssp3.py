from __future__ import annotations

from typing import Callable
from typing import Dict
from typing import Tuple

import torch
from peft import TaskType
from transformers import EvalPrediction

from protbench import metrics
from protbench.applications.benchmarking_task import BenchmarkingTask
from protbench.models.heads import TokenClassificationHead
from protbench.tasks import HuggingFaceResidueToClass
from protbench.utils.preprocessing_utils import (
    preprocess_multi_classification_logits,
)


def preprocess_ssp_rows(seq, label, mask):
    mask = list(map(float, mask.split()))
    return seq, label, mask


def get_ssp3_casp12_dataset():
    train_data = HuggingFaceResidueToClass(
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="training_hhblits.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp3",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )

    val_data = HuggingFaceResidueToClass(
        class_to_id=train_data.class_to_id,
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="CASP12.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp3",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )
    return train_data, val_data


def get_ssp3_casp14_dataset():
    train_data = HuggingFaceResidueToClass(
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="training_hhblits.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp3",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )
    val_data = HuggingFaceResidueToClass(
        class_to_id=train_data.class_to_id,
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="CASP14.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp3",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )

    return train_data, val_data


def get_ssp3_cb513_dataset():
    train_data = HuggingFaceResidueToClass(
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="training_hhblits.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp3",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )
    val_data = HuggingFaceResidueToClass(
        class_to_id=train_data.class_to_id,
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="CB513.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp3",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )
    return train_data, val_data


def get_ssp3_ts115_dataset():
    train_data = HuggingFaceResidueToClass(
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="training_hhblits.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp3",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )
    val_data = HuggingFaceResidueToClass(
        class_to_id=train_data.class_to_id,
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="TS115.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp3",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )
    return train_data, val_data


supported_datasets = {
    "ssp3_casp12": get_ssp3_casp12_dataset,
    "ssp3_casp14": get_ssp3_casp14_dataset,
    "ssp3_ts115": get_ssp3_ts115_dataset,
    "ssp3_cb513": get_ssp3_cb513_dataset,
}


def compute_secondary_structure_metrics(p: EvalPrediction) -> Dict:
    accuracies_std = metrics.compute_accuracies_std(p)
    num_examples = p.label_ids.shape[0]
    error_bar = metrics.compute_accuracies_error_bar(
        accuracies_std=accuracies_std, num_examples=num_examples
    )
    return {
        "accuracy": metrics.compute_accuracy(p),
        "precision": metrics.compute_precision(p, average="macro"),
        "recall": metrics.compute_recall(p, average="macro"),
        "f1": metrics.compute_f1(p, average="macro"),
        "accuracy_std": accuracies_std,
        "error_bar": error_bar,
    }


class SSP3(BenchmarkingTask):
    task_type = TaskType.TOKEN_CLS
    requires_pooling = False

    def __init__(
        self,
        dataset: str,
        from_embeddings: bool = False,
        tokenizer: Callable | None = None,
    ):
        train_dataset, eval_dataset = supported_datasets[dataset]()
        super().__init__(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            preprocessing_fn=preprocess_multi_classification_logits,
            metrics_fn=compute_secondary_structure_metrics,
            metric_for_best_model="eval_validation_accuracy",
            from_embeddings=from_embeddings,
            tokenizer=tokenizer,
        )

    def get_train_data(self) -> Tuple:
        return self.train_dataset.data[0], self.train_dataset.data[1]

    def get_eval_data(self) -> Tuple:
        return self.eval_dataset.data[0], self.eval_dataset.data[1]

    def get_task_head(self, embedding_dim) -> torch.nn.Module:
        head = TokenClassificationHead(
            input_dim=embedding_dim, output_dim=self.get_num_classes()
        )
        return head
