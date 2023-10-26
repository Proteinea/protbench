import abc


class BenchmarkingTask(abc.ABC):
    def __init__(
        self,
        train_dataset,
        eval_dataset,
        preprocessing_fn,
        collate_fn,
        metrics_fn,
        metric_for_best_model=None,
        from_embeddings=False,
        tokenizer=None,
        requires_pooling=False,
        task_type=None,
    ):
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
