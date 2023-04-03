import abc


class Task(abc.ABC):
    @abc.abstractmethod
    def load_sequences(self):
        raise NotImplementedError(
            "`load_sequences` is not implemented "
            "and should be implemented in the subclass."
        )

    @abc.abstractmethod
    def load_labels(self):
        raise NotImplementedError(
            "`load_labels` is not implemented "
            "and should be implemented in the subclass."
        )
