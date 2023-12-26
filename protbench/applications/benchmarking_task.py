import abc
from typing import Callable

from peft import TaskType
from protbench.tasks.task import Task


class BenchmarkingTask(abc.ABC):
    def __init__(
        self,
        train_dataset: Task,
        eval_dataset: Task,
        preprocessing_fn: Callable,
        collate_fn: Callable,
        metrics_fn: Callable,
        metric_for_best_model: str = None,
        from_embeddings: bool = False,
        tokenizer: Callable = None,
        requires_pooling: bool = False,
        task_type: TaskType = None,
    ):
        """Base class for downstream tasks.

        Args:
            train_dataset (Task): A task instance that contains training data.
            eval_dataset (Task): A task instance that contains validation data.
            preprocessing_fn (Callable): Preprocessing function that will be
                                         called to prepare the outputs before
                                         passing them to the metrics function.
            collate_fn (Callable): Collate function that will be passed to the
                                   dataloader to prepare the batch of inputs
                                   before passing them to the model.
            metrics_fn (Callable): Metrics function that will be called after
                                   each N steps.
            metric_for_best_model (str, optional): Metric that will be used to
                                                   pick the best model.
                                                   Defaults to None.
            from_embeddings (bool, optional): Whether the inputs to the model
                                              are extracted embeddings or not.
                                              Defaults to False.
            tokenizer (Callable, optional): Tokenizer that will be used to
                                            encode the inputs, if
                                            `from_embeddings` is set to `True`
                                            then no need for the tokenizer.
                                            Defaults to None.
            requires_pooling (bool, optional): Whether the specified task
                                               requires pooling layer or not.
                                               Defaults to False.
            task_type (TaskType, optional): If LoRA or any parameter efficient
                                            finetuning is going to be used then
                                            TaskType could be stored here.
                                            Defaults to None.

        Raises:
            ValueError: If `from_embedding` is set to `False` and tokenizer is
                        set to `None`.
        """
        if not from_embeddings and tokenizer is None:
            raise ValueError(
                "Expected a `tokenizer`  when `from_embeddings` "
                f"is set to `False`. Received: {tokenizer}."
            )
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
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

    @abc.abstractmethod
    def get_eval_data(self):
        raise NotImplementedError("Should be implemented in a subclass.")

    @abc.abstractmethod
    def get_downstream_model(self):
        raise NotImplementedError("Should be implemented in a subclass.")

    def get_num_classes(self):
        return getattr(self.train_dataset, "num_classes", None)
