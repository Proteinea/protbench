from typing import Callable, Optional

from peft import TaskType
from protbench import metrics
from protbench.applications.benchmarking_task import BenchmarkingTask
from protbench.models.downstream_models import (
    DownstreamModelFromEmbedding, DownstreamModelWithPretrainedBackbone)
from protbench.models.heads import RegressionHead
from protbench.tasks import HuggingFaceSequenceToValue
from protbench.utils import collate_inputs, collate_sequence_and_labels
from transformers import EvalPrediction


def get_fluorescence_dataset():
    train_data = HuggingFaceSequenceToValue(
        dataset_url="proteinea/fluorosence",
        data_files=None,
        data_key="train",
        seqs_col="primary",
        labels_col="log_fluorescence",
    )
    val_data = HuggingFaceSequenceToValue(
        dataset_url="proteinea/fluorosence",
        data_files=None,
        data_key="test",
        seqs_col="primary",
        labels_col="log_fluorescence",
    )
    return train_data, val_data


supported_datasets = {"fluorescence": get_fluorescence_dataset}


def compute_fluoresscence_metrics(p: EvalPrediction):
    return {
        "spearman": metrics.compute_spearman(p),
    }


class Fluorescence(BenchmarkingTask):
    def __init__(
        self,
        dataset: str = "fluorescence",
        from_embeddings: bool = False,
        tokenizer: Optional[Callable] = None,
        task_type: Optional[TaskType] = None,
    ):
        train_dataset, eval_dataset = supported_datasets[dataset]()
        collate_fn = (
            collate_inputs
            if from_embeddings
            else collate_sequence_and_labels(tokenizer=tokenizer)
        )

        super().__init__(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            preprocessing_fn=None,
            collate_fn=collate_fn,
            metrics_fn=compute_fluoresscence_metrics,
            metric_for_best_model="spearman",
            from_embeddings=from_embeddings,
            tokenizer=tokenizer,
            requires_pooling=True,
            task_type=task_type,
        )

    def get_train_data(self):
        return self.train_dataset.data[0], self.train_dataset.data[1]

    def get_eval_data(self):
        return self.eval_dataset.data[0], self.eval_dataset.data[1]

    def get_downstream_model(self, backbone_model, embedding_dim, pooling=None):
        head = RegressionHead(input_dim=embedding_dim)
        if self.from_embeddings:
            model = DownstreamModelFromEmbedding(backbone_model, head)
        else:
            model = DownstreamModelWithPretrainedBackbone(
                backbone_model, head, pooling
            )
        return model
