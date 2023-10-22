import os

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, labels, shift_left=0, shift_right=1):
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
        self.shift_left = shift_left
        self.shift_right = shift_right

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embds = self.embeddings[idx][self.shift_left : -self.shift_right, :]
        labels = torch.tensor(self.labels[idx])
        return {
            "embds": embds,
            "labels": labels,
        }


class EmbeddingsDatasetFromDisk(Dataset):
    def __init__(self, embeddings_path, labels, shift_left=0, shift_right=1):
        """Dataset for embeddings and corresponding labels of a task.

        Args:
            embeddings (list[torch.Tensor]): list of tensors of embeddings (batch_size, seq_len, embd_dim)
                where each tensor may have a different seq_len.
            labels (list[Any]): list of labels.
        """
        if len(os.listdir(embeddings_path)) != len(labels):
            raise ValueError(
                "embeddings and labels must have the same length but got "
                f"{len(embeddings_path)} and {len(labels)}"
            )
        self.embeddings = embeddings_path
        self.labels = labels
        self.shift_left = shift_left
        self.shift_right = shift_right

    def __len__(self):
        return len(os.listdir(self.embeddings))

    def __getitem__(self, idx):
        embedding_path = os.path.join(self.embeddings, f"{idx}.npy")
        embds = torch.from_numpy(
            np.load(embedding_path)[self.shift_left : -self.shift_right, :]
        )
        labels = torch.tensor(self.labels[idx])
        return {
            "embds": embds,
            "labels": labels,
        }


class SequenceAndLabelsDataset(Dataset):
    def __init__(self, sequences, labels) -> None:
        super().__init__()
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        assert len(self.sequences) == len(self.labels)
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "sequences": self.sequences[idx],
            "labels": self.labels[idx],
        }
