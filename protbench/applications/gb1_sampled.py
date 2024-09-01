from __future__ import annotations

from typing import Callable
from typing import Dict
from typing import Tuple

import torch
from peft import TaskType
from transformers import EvalPrediction

from protbench import metrics
from protbench.applications.benchmarking_task import BenchmarkingTask
from protbench.models.heads import RegressionHead
from protbench.tasks import HuggingFaceSequenceToValue
from protbench.utils import collate_inputs
from protbench.utils import collate_sequence_and_labels


def load_gb1_dataset():
    train_data = HuggingFaceSequenceToValue(
        dataset_url="proteinea/gb1_sampled",
        data_files="gb1-sampled-train.csv",
        data_key="train",
        seqs_col="sequence",
        labels_col="target",
    )
    val_data = HuggingFaceSequenceToValue(
        dataset_url="proteinea/gb1_sampled",
        data_files="gb1-sampled-eval.csv",
        data_key="train",
        seqs_col="sequence",
        labels_col="target",
    )
    test_data = HuggingFaceSequenceToValue(
        dataset_url="proteinea/gb1_sampled",
        data_files="gb1-sampled-test.csv",
        data_key="train",
        seqs_col="sequence",
        labels_col="target",
    )
    return train_data, val_data, test_data


supported_datasets = {
    "gb1_sampled": load_gb1_dataset,
}


def compute_gb1_metrics(p: EvalPrediction) -> Dict:
    spearman = metrics.compute_spearman(p)
    num_examples = p.label_ids.shape[0]
    error_bar = metrics.compute_error_bar_for_regression(
        spearman_corr=spearman, num_examples=num_examples
    )
    return {
        "spearman": spearman,
        "error_bar": error_bar,
    }


class GB1Sampled(BenchmarkingTask):
    task_type = TaskType.SEQ_CLS
    requires_pooling = True

    def __init__(
        self,
        dataset: str = "gb1_sampled",
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
            metrics_fn=compute_gb1_metrics,
            metric_for_best_model="eval_validation_spearman",
            from_embeddings=from_embeddings,
            tokenizer=tokenizer,
            collate_fn=collate_fn,
        )

    def load_train_data(self) -> Tuple:
        return self.train_dataset.data[0], self.train_dataset.data[1]

    def load_eval_data(self) -> Tuple:
        return self.eval_dataset.data[0], self.eval_dataset.data[1]

    def load_test_data(self) -> Tuple:
        return self.test_dataset.data[0], self.test_dataset.data[1]

    def load_task_head(self, embedding_dim) -> torch.nn.Module:
        head = RegressionHead(input_dim=embedding_dim)
        return head
