import abc
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
from protbench.embedder import EmbeddingFunction


class Embedder(abc.ABC):
    def __init__(
        self,
        embedding_function: EmbeddingFunction,
        low_memory: bool = False,
        save_path: Optional[str] = None,
    ):
        self.embedding_function = embedding_function
        if low_memory and save_path is None:
            raise ValueError(
                "Expected save_path to be set when low_memory is True"
            )
        self.low_memory = low_memory
        if save_path:
            self.save_path = Path(save_path)
            self.save_path.mkdir(exist_ok=True, parents=True)
        else:
            self.save_path = None

    @staticmethod
    def save_embedding_to_disk(
        idx: int, embedding: np.ndarray, save_path: Path
    ) -> None:
        np.save(save_path / Path(str(idx)), embedding)

    @abc.abstractmethod
    def run(self, sequences: Iterable[str]) -> List[Any]:
        raise NotImplementedError
