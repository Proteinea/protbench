from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm.auto import tqdm

from protbench.src.embedder import EmbeddingFunction


class Embedder:
    def __init__(self, embedding_function: EmbeddingFunction, save_path: str = None):
        self.embedding_function = embedding_function
        self.save_path = Path(save_path) or None
        if self.save_path is None:
            self.embeddings = []
        else:
            self.save_path.mkdir(exist_ok=True)

    def save_to_disk_or_store_in_memory(self, idx, embedding):
        if self.save_path is not None:
            np.save(
                self.save_path / Path(str(idx)),
                self.embedding_function.to_numpy(embedding),
            )
        else:
            self.embeddings.append(embedding)

    def export_embeddings(self, path: str) -> None:
        for idx, embedding in enumerate(self.embeddings):
            np.save(path / Path(str(idx)), self.embedding_function.to_numpy(embedding))

    def embed_data(self, sequences: Iterable[str]) -> None:
        for idx, example in enumerate(tqdm(sequences)):
            embedding = self.embedding_function(example)
            self.save_to_disk_or_store_in_memory(idx, embedding)
