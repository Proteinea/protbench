from __future__ import annotations

from typing import Callable
from typing import Dict
from typing import Tuple

import torch
from peft import TaskType

from protbench import metrics
from protbench.applications.benchmarking_task import BenchmarkingTask
from protbench.models.heads import RegressionHead
from protbench.tasks import HuggingFaceSequenceToValue
from protbench.utils import collate_inputs
from protbench.utils import collate_sequence_and_labels


def get_thermostability_dataset():
    train_data = HuggingFaceSequenceToValue(
        dataset_url="proteinea/thermostability",
        data_files="thermo_train.csv",
        data_key="train",
        seqs_col="sequence",
        labels_col="target",
    )
    val_data = HuggingFaceSequenceToValue(
        dataset_url="proteinea/thermostability",
        data_files="thermo_eval.csv",
        data_key="train",
        seqs_col="sequence",
        labels_col="target",
    )
    test_data = HuggingFaceSequenceToValue(
        dataset_url="proteinea/thermostability",
        data_files="thermo_test.csv",
        data_key="train",
        seqs_col="sequence",
        labels_col="target",
    )
    return train_data, val_data, test_data


supported_datasets = {
    "thermostability": get_thermostability_dataset,
}


def compute_thermostability_metrics(p) -> Dict:
    spearman = metrics.compute_spearman(p)
    num_examples = p.label_ids.shape[0]
    error_bar = metrics.compute_error_bar_for_regression(
        spearman_corr=spearman, num_examples=num_examples
    )
    return {
        "spearman": spearman,
        "error_bar": error_bar,
    }


class Thermostability(BenchmarkingTask):
    task_type = TaskType.SEQ_CLS
    requires_pooling = True

    def __init__(
        self,
        dataset: str = "thermostability",
        from_embeddings: bool = False,
        tokenizer: Callable | None = None,
    ):
        train_dataset, eval_dataset, test_dataset = supported_datasets[
            dataset
        ]()
        if from_embeddings:
            collate_fn = collate_inputs
        elif tokenizer is not None:
            collate_fn = collate_sequence_and_labels(tokenizer)
        else:
            raise ValueError(
                "Expected a `tokenizer`  when `from_embeddings` "
                f"is set to `False`. Received: {tokenizer}."
            )

        super().__init__(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            test_dataset=test_dataset,
            preprocessing_fn=None,
            metrics_fn=compute_thermostability_metrics,
            metric_for_best_model="eval_validation_spearman",
            from_embeddings=from_embeddings,
            tokenizer=tokenizer,
            collate_fn=collate_fn,
        )

    def get_train_data(self) -> Tuple:
        return self.train_dataset.data[0], self.train_dataset.data[1]

    def get_eval_data(self) -> Tuple:
        return self.eval_dataset.data[0], self.eval_dataset.data[1]

    def get_test_data(self) -> Tuple:
        return self.test_dataset.data[0], self.test_dataset.data[1]

    def get_task_head(self, embedding_dim) -> torch.nn.Module:
        head = RegressionHead(input_dim=embedding_dim)
        return head
