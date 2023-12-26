from typing import Any

import torch
from torch.utils.data import Dataset


class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings: list[torch.Tensor], labels: list[Any]):
        """Dataset for embeddings and corresponding labels of a task.

        Args:
            embeddings (list[torch.Tensor]): list of tensors of embeddings (batch_size, seq_len, embd_dim)
                where each tensor may have a different seq_len.
            labels (list[Any]): list of labels.
        """
        if len(embeddings) != len(labels):
            raise ValueError(
                "embeddings and labels must have the same length but got "
                f"{len(embeddings)} and {len(labels)}"
            )
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "embd": self.embeddings[idx],
            "labels": torch.tensor(self.labels[idx]),
        }
