import abc
from typing import Any
from typing import Callable
from typing import List
from typing import Union


class EmbeddingFunction(abc.ABC):
    def __init__(self, model: Callable, tokenization_fn: Callable):
        """Abstract class for embedding functions.

        Args:
            model (Callable): model to use for embedding.
            tokenizer (Callable): tokenizer to use for embedding.
        """
        self.model = model
        self.tokenization_fn = tokenization_fn

    @abc.abstractmethod
    def call(
        self, sequence: Union[str, List[str]], *args, **kwargs
    ) -> Union[Any, List[Any]]:
        # Embed a sequence or a list of sequences. If a single sequence is
        # passed, a single embedding is returned.
        # If a list of sequences is passed, a list of embeddings is returned.
        raise NotImplementedError

    def __call__(
        self, sequence: Union[str, List[str]], *args, **kwargs
    ) -> Union[Any, List[Any]]:
        if not (isinstance(sequence, str) or isinstance(sequence, list)):
            raise TypeError(
                f"Expected sequence to be of type str, got {type(sequence)}"
            )
        return self.call(sequence)
