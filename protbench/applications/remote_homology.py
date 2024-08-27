from __future__ import annotations

from functools import partial
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np
import torch
from peft import TaskType
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import top_k_accuracy_score

from protbench.applications.benchmarking_task import BenchmarkingTask
from protbench.models.heads import MultiClassClassificationHead
from protbench.tasks import HuggingFaceSequenceToClass
from protbench.utils import collate_inputs
from protbench.utils import collate_sequence_and_labels


def load_remote_homology_dataset():
    train_data = HuggingFaceSequenceToClass(
        dataset_url="proteinea/remote_homology",
        seqs_col="primary",
        labels_col="fold_label",
        data_files=None,
        data_key="train",
    )

    val_data = HuggingFaceSequenceToClass(
        class_to_id=train_data.class_to_id,
        dataset_url="proteinea/remote_homology",
        seqs_col="primary",
        labels_col="fold_label",
        data_files=None,
        data_key="validation",
    )

    test_data = HuggingFaceSequenceToClass(
        class_to_id=train_data.class_to_id,
        dataset_url="proteinea/remote_homology",
        seqs_col="primary",
        labels_col="fold_label",
        data_files="test_fold_holdout.csv",
        data_key="train",
    )
    return train_data, val_data, test_data


supported_datasets = {
    "remote_homology": load_remote_homology_dataset,
}


def compute_remote_homology_metrics(p, num_classes) -> Dict:
    prfs = precision_recall_fscore_support(
        p.label_ids, p.predictions.argmax(axis=1), average="macro"
    )

    return {
        "accuracy": accuracy_score(p.label_ids, p.predictions.argmax(axis=1)),
        "precision": prfs[0],
        "recall": prfs[1],
        "f1": prfs[2],
        "hits10": top_k_accuracy_score(
            p.label_ids, p.predictions, k=10, labels=np.arange(num_classes)
        ),
    }


class RemoteHomology(BenchmarkingTask):
    task_type = TaskType.SEQ_CLS
    requires_pooling = True

    def __init__(
        self,
        dataset: str = "remote_homology",
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
            preprocessing_fn=None,
            metrics_fn=partial(
                compute_remote_homology_metrics,
                num_classes=train_dataset.num_classes,
            ),
            metric_for_best_model="eval_validation_hits10",
            from_embeddings=from_embeddings,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
            collate_fn=collate_fn,
        )

    def load_train_data(self) -> Tuple:
        return self.train_dataset.data[0], self.train_dataset.data[1]

    def load_eval_data(self) -> Tuple:
        return self.eval_dataset.data[0], self.eval_dataset.data[1]

    def load_test_data(self) -> Tuple:
        return self.test_dataset.data[0], self.test_dataset.data[1]

    def load_task_head(self, embedding_dim) -> torch.nn.Module:
        head = MultiClassClassificationHead(
            input_dim=embedding_dim, output_dim=self.num_classes
        )
        return head
