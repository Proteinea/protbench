from __future__ import annotations

from typing import Callable
from typing import Dict
from typing import Tuple

import torch
from peft import TaskType
from transformers import EvalPrediction

from protbench import metrics
from protbench.applications.benchmarking_task import BenchmarkingTask
from protbench.models.heads import BinaryClassificationHead
from protbench.tasks import HuggingFaceSequenceToClass
from protbench.utils import collate_inputs
from protbench.utils import collate_sequence_and_labels
from protbench.utils import preprocess_binary_classification_logits


def load_solubility_dataset():
    train_data = HuggingFaceSequenceToClass(
        dataset_url="proteinea/solubility",
        data_files=None,
        data_key="train",
        seqs_col="sequences",
        labels_col="labels",
    )
    validation_data = HuggingFaceSequenceToClass(
        class_to_id=train_data.class_to_id,
        dataset_url="proteinea/solubility",
        data_files=None,
        data_key="validation",
        seqs_col="sequences",
        labels_col="labels",
    )
    test_data = HuggingFaceSequenceToClass(
        class_to_id=train_data.class_to_id,
        dataset_url="proteinea/solubility",
        data_files=None,
        data_key="test",
        seqs_col="sequences",
        labels_col="labels",
    )
    return train_data, validation_data, test_data


supported_datasets = {
    "solubility": load_solubility_dataset,
}


def compute_solubility_metrics(p: EvalPrediction) -> Dict:
    accuracies_std = metrics.compute_accuracies_std(p)
    num_examples = p.label_ids.shape[0]
    error_bar = metrics.compute_accuracies_error_bar(
        accuracies_std=accuracies_std, num_examples=num_examples
    )
    return {
        "accuracy": metrics.compute_accuracy(p),
        "precision": metrics.compute_precision(p, average="binary"),
        "recall": metrics.compute_recall(p, average="binary"),
        "f1": metrics.compute_f1(p, average="binary"),
        "accuracy_std": accuracies_std,
        "error_bar": error_bar,
    }


class Solubility(BenchmarkingTask):
    task_type = TaskType.SEQ_CLS
    requires_pooling = True

    def __init__(
        self,
        dataset: str = "solubility",
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
            preprocessing_fn=preprocess_binary_classification_logits,
            metrics_fn=compute_solubility_metrics,
            metric_for_best_model="eval_validation_accuracy",
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
        head = BinaryClassificationHead(input_dim=embedding_dim)
        return head
