from __future__ import annotations

import abc
from typing import Callable

from protbench.models.downstream_models import DownstreamModelFromEmbedding
from protbench.models.downstream_models import (
    DownstreamModelWithPretrainedBackbone,
)
from protbench.tasks.task import Task
from protbench.utils import collate_inputs
from protbench.utils import collate_sequence_and_labels


class BenchmarkingTask(abc.ABC):
    task_type = None
    requires_pooling = None

    def __init__(
        self,
        train_dataset: Task,
        eval_dataset: Task,
        preprocessing_fn: Callable,
        metrics_fn: Callable,
        test_dataset: Task | None = None,
        metric_for_best_model: str = None,
        from_embeddings: bool = False,
        tokenizer: Callable = None,
    ):
        """Base class for benchmarking datasets.

        Args:
            train_dataset (Task): Training dataset.
            eval_dataset (Task): Validation dataset.
            preprocessing_fn (Callable): Preprocessing function that will
                be called on the logits during evaluation.
            metrics_fn (Callable): Metrics function that will be
                called for evaluation.
            test_dataset (Task | None, optional): Test dataset.
                Defaults to None.
            metric_for_best_model (str, optional): metric that will be used to
                save/upload the best model. Defaults to None.
            from_embeddings (bool, optional): Whether the downstream model
                should expect embeddings or input ids. Defaults to False.
            tokenizer (Callable, optional): Tokenization function.
                Defaults to None.

        Raises:
            ValueError: If `tokenizer` is None while `from_embeddings` is False.
        """
        if self.from_embeddings:
            self.collate_fn = collate_inputs
        elif tokenizer is not None:
            self.collate_fn = collate_sequence_and_labels(tokenizer)
        else:
            raise ValueError(
                "Expected a `tokenizer`  when `from_embeddings` "
                f"is set to `False`. Received: {tokenizer}."
            )

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.preprocessing_fn = preprocessing_fn
        self.metrics_fn = metrics_fn
        self.metric_for_best_model = metric_for_best_model
        self.from_embeddings = from_embeddings
        self.tokenizer = tokenizer

    @abc.abstractmethod
    def get_train_data(self):
        raise NotImplementedError("Should be implemented in a subclass.")

    def get_eval_data(self):
        raise NotImplementedError("Should be implemented in a subclass.")

    def get_test_data(self):
        raise NotImplementedError("Should be implemented in a subclass.")

    @abc.abstractmethod
    def get_task_head(self, embedding_dim):
        raise NotImplementedError("Should be implemented in a subclass.")

    @abc.abstractmethod
    def get_downstream_model(
        self, backbone_model, embedding_dim, pooling=None
    ):
        head = self.get_task_head(embedding_dim=embedding_dim)
        if self.from_embeddings:
            model = DownstreamModelFromEmbedding(backbone_model, head)
        else:
            model = DownstreamModelWithPretrainedBackbone(
                backbone_model, head, pooling
            )
        return model

    def get_num_classes(self):
        return getattr(self.train_dataset, "num_classes", None)
