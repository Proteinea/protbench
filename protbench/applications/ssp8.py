from typing import Any, Callable, List, Optional, Union

import numpy as np
from peft import TaskType
from protbench import metrics
from protbench.applications.benchmarking_task import BenchmarkingTask
from protbench.models.downstream_models import (
    DownstreamModelFromEmbedding, DownstreamModelWithPretrainedBackbone)
from protbench.models.heads import TokenClassificationHead
from protbench.tasks import HuggingFaceResidueToClass
from protbench.utils import (collate_inputs_and_labels,
                             collate_sequence_and_align_labels)
from protbench.utils.preprocessing_utils import \
    preprocess_multi_classification_logits
from transformers import EvalPrediction


def preprocess_ssp_rows(seq: str, label: Union[List, np.ndarray], mask: Any):
    mask = list(map(float, mask.split()))
    return seq, label, mask


def get_ssp8_casp12_dataset():
    train_data = HuggingFaceResidueToClass(
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="training_hhblits.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp8",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )

    val_data = HuggingFaceResidueToClass(
        class_to_id=train_data.class_to_id,
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="CASP12.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp8",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )
    return train_data, val_data


def get_ssp8_casp14_dataset():
    train_data = HuggingFaceResidueToClass(
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="training_hhblits.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp8",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )
    val_data = HuggingFaceResidueToClass(
        class_to_id=train_data.class_to_id,
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="CASP14.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp8",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )

    return train_data, val_data


def get_ssp8_cb513_dataset():
    train_data = HuggingFaceResidueToClass(
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="training_hhblits.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp8",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )
    val_data = HuggingFaceResidueToClass(
        class_to_id=train_data.class_to_id,
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="CB513.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp8",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )
    return train_data, val_data


def get_ssp8_ts115_dataset():
    train_data = HuggingFaceResidueToClass(
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="training_hhblits.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp8",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )
    val_data = HuggingFaceResidueToClass(
        class_to_id=train_data.class_to_id,
        dataset_url="proteinea/secondary_structure_prediction",
        data_files="TS115.csv",
        data_key="train",
        seqs_col="input",
        labels_col="dssp8",
        mask_col="disorder",
        preprocessing_function=preprocess_ssp_rows,
    )
    return train_data, val_data


supported_datasets = {
    "ssp8_casp12": get_ssp8_casp12_dataset,
    "ssp8_casp14": get_ssp8_casp14_dataset,
    "ssp8_ts115": get_ssp8_ts115_dataset,
    "ssp8_cb513": get_ssp8_cb513_dataset,
}


def compute_secondary_structure_metrics(p: EvalPrediction):
    return {
        "accuracy": metrics.compute_accuracy(p),
        "precision": metrics.compute_precision(p, average="macro"),
        "recall": metrics.compute_recall(p, average="macro"),
        "f1": metrics.compute_f1(p, average="macro"),
    }


class SSP8(BenchmarkingTask):
    def __init__(
        self,
        dataset: str,
        from_embeddings: bool = False,
        tokenizer: Optional[Callable] = None,
        task_type: Optional[TaskType] = None,
    ):
        train_dataset, eval_dataset = supported_datasets[dataset]()
        collate_fn = (
            collate_inputs_and_labels
            if from_embeddings
            else collate_sequence_and_align_labels(tokenizer)
        )
        super().__init__(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            preprocessing_fn=preprocess_multi_classification_logits,
            collate_fn=collate_fn,
            metrics_fn=compute_secondary_structure_metrics,
            metric_for_best_model="accuracy",
            from_embeddings=from_embeddings,
            tokenizer=tokenizer,
            requires_pooling=False,
            task_type=task_type,
        )

    def get_train_data(self):
        return self.train_dataset.data[0], self.train_dataset.data[1]

    def get_eval_data(self):
        return self.eval_dataset.data[0], self.eval_dataset.data[1]

    def get_downstream_model(self, backbone_model, embedding_dim, pooling=None):
        head = TokenClassificationHead(
            input_dim=embedding_dim, output_dim=self.get_num_classes()
        )
        if self.from_embeddings:
            model = DownstreamModelFromEmbedding(backbone_model, head)
        else:
            model = DownstreamModelWithPretrainedBackbone(backbone_model, head)
        return model

    def initialize_metrics(self):
        return lambda x: {
            "accuracy": metrics.compute_accuracy(x),
            "precision": metrics.compute_precision(x, average="macro"),
            "recall": metrics.compute_recall(x, average="macro"),
            "f1": metrics.compute_f1(x, average="macro"),
        }
