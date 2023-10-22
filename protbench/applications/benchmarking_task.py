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
    ):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.preprocessing_fn = preprocessing_fn
        self.collate_fn = collate_fn
        self.metrics_fn = metrics_fn
        self.metric_for_best_model = metric_for_best_model
        self.from_embeddings = from_embeddings
        self.tokenizer = tokenizer

    def get_train_data(self):
        raise NotImplementedError("Should be implemented in a subclass.")

    def get_eval_data(self):
        raise NotImplementedError("Should be implemented in a subclass.")

    def get_downstream_model(self):
        raise NotImplementedError("Should be implemented in a subclass.")

    def get_num_classes(self):
        return getattr(self.train_dataset, "num_classes", None)
