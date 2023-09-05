import logging
from pathlib import Path
from threading import Thread
from typing import Iterable, Literal, List

import numpy as np
from tqdm.auto import tqdm

from protbench.src.embedder import EmbeddingFunction


class Embedder:
    def __init__(
        self,
        embedding_function: EmbeddingFunction,
        save_to: Literal["memory", "disk", "both"] = None,
        save_path: str | None = None,
        buffer_size: int = 1000,
    ):
        self.embedding_function = embedding_function
        if save_to == "memory":
            self.save_path = None
            self.embeddings = []
        elif save_to == "disk":
            self.save_path = Path(save_path) if save_path else Path("embeddings")
            self.save_path.mkdir(exist_ok=True, parents=True)
            self.embeddings = []
        elif save_to == "both":
            self.save_path = Path(save_path) if save_path else Path("embeddings")
            self.save_path.mkdir(exist_ok=True, parents=True)
            self.embeddings = []
        else:
            raise ValueError(
                f"Expected save_to to be one of ['memory', 'disk', 'both'], got {save_to}"
            )
        self.save_to = save_to
        self.buffer_size = buffer_size
        self.thread = None

    def _prepare_embeddings_for_thread(self, start_idx: int, embeddings):
        self.thread.join() if self.thread else None
        embeddings_np = [self.embedding_function.to_numpy(e) for e in embeddings]
        self.thread = Thread(target=self.save_to_disk, args=(start_idx, embeddings_np))

    def save_to_disk_or_store_in_memory(self, idx, embedding):
        if self.save_to == "memory" or self.save_to == "both":
            self.embeddings.append(embedding)
        elif self.save_to == "disk":
            self.embeddings.append(embedding)
            if idx % self.buffer_size == 0:
                logging.info(f"Saving embeddings {idx - self.buffer_size + 1} to {idx}")
                self._prepare_embeddings_for_thread(idx, self.embeddings)
                self.thread.start()
                self.embeddings = []
        else:
            if idx % self.buffer_size == 0:
                logging.info(f"Saving embeddings {idx - self.buffer_size + 1} to {idx}")
                self._prepare_embeddings_for_thread(
                    idx, self.embeddings[idx - self.buffer_size + 1 : idx + 1]
                )
                self.thread.start()
            self.embeddings.append(embedding)

    def save_to_disk(self, start_index: int, embeddings: List[np.ndarray]) -> None:
        for idx, embedding in enumerate(embeddings):
            np.save(self.save_path / Path(str(start_index + idx)), embedding)

    def embed_data(self, sequences: Iterable[str]) -> None:
        for idx, example in enumerate(
            tqdm(sequences, desc="Embedding Sequences", unit=" sequences")
        ):
            embedding = self.embedding_function(example)
            self.save_to_disk_or_store_in_memory(idx, embedding)
