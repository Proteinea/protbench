import abc
from typing import Callable
from protbench.models.downstream_models import DownstreamModelFromEmbedding
from protbench.models.downstream_models import \
    DownstreamModelWithPretrainedBackbone
from protbench.tasks.task import Task
from typing import Optional
from peft import TaskType


class BenchmarkingTask(abc.ABC):
    def __init__(
        self,
        train_dataset: Task,
        eval_dataset: Task,
        preprocessing_fn: Callable,
        collate_fn: Callable,
        metrics_fn: Callable,
        test_dataset: Optional[Task] = None,
        metric_for_best_model: str = None,
        from_embeddings: bool = False,
        tokenizer: Callable = None,
        requires_pooling: bool = False,
        task_type: TaskType = None,
    ):
        """Base class for benchmarking datasets.

        Args:
            train_dataset (Task): Training dataset.
            eval_dataset (Task): Validation dataset.
            preprocessing_fn (Callable): Preprocessing function that will
                be called on the logits during evaluation.
            collate_fn (Callable): Collate function that will be called
                after collecting the batch.
            metrics_fn (Callable): Metrics function that will be
                called for evaluation.
            test_dataset (Optional[Task], optional): Test dataset.
                Defaults to None.
            metric_for_best_model (str, optional): metric that will be used to
                save/upload the best model. Defaults to None.
            from_embeddings (bool, optional): Whether the downstream model
                should expect embeddings or input ids. Defaults to False.
            tokenizer (Callable, optional): Tokenization function.
                Defaults to None.
            requires_pooling (bool, optional): Whether this task requires
                pooling for the embeddings before passing it to
                the logits or not. Defaults to False.
            task_type (TaskType, optional): Task type that will be used if
                using PEFT. Defaults to None.

        Raises:
            ValueError: If `tokenizer` is None while `from_embeddings` is False.
        """
        if not from_embeddings and tokenizer is None:
            raise ValueError(
                "Expected a `tokenizer`  when `from_embeddings` "
                f"is set to `False`. Received: {tokenizer}."
            )
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.preprocessing_fn = preprocessing_fn
        self.collate_fn = collate_fn
        self.metrics_fn = metrics_fn
        self.metric_for_best_model = metric_for_best_model
        self.from_embeddings = from_embeddings
        self.tokenizer = tokenizer
        self.requires_pooling = requires_pooling
        self.task_type = task_type

    @abc.abstractmethod
    def get_train_data(self):
        raise NotImplementedError("Should be implemented in a subclass.")

    def get_eval_data(self):
        raise NotImplementedError("Should be implemented in a subclass.")

    def get_test_data(self):
        raise NotImplementedError("Should be implemented in a subclass.")

    def get_task_head(self, embedding_dim):
        raise NotImplementedError("Should be implemented in a subclass.")

    @abc.abstractmethod
    def get_downstream_model(self, backbone_model, embedding_dim, pooling=None):
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
