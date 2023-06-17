from typing import List

from torch.utils.data import Dataset


class SequencesDataset(Dataset):
    def __init__(self, sequences: List[str]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]
