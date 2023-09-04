import numpy as np


import abc
from typing import Any


class EmbeddingFunction(abc.ABC):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @abc.abstractmethod
    def call(self, sequence: str) -> Any:
        raise NotImplementedError

    @abc.abstractclassmethod
    def to_numpy(cls, tensor: Any) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, sequence: str) -> Any:
        if not isinstance(sequence, str):
            raise TypeError(
                f"Expected sequence to be of type str, got {type(sequence)}"
            )
        return self.call(sequence)