from os import PathLike
from typing import List
from typing import Tuple

from Bio import SeqIO

from protbench.tasks.sequence_to_value.sequence_to_value import SequenceToValue


class FastaSequenceToValue(SequenceToValue):
    def __init__(self, data_file: PathLike) -> None:
        """Generic task of predicting a value for a sequence.

        Args:
            data_file (PathLike): Path to the fasta file containing
                the sequences and labels. The file must have the
                following format:
                >seq_id VALUE=value
                sequence

        """
        super(FastaSequenceToValue, self).__init__()

        self._data = self.load_and_preprocess_data(data_file)

    @property
    def data(self) -> Tuple[List[str], List[float]]:
        return self._data

    def load_and_preprocess_data(self, data_file) -> None:
        seqs, labels = [], []
        for item in SeqIO.parse(data_file, "fasta"):
            label = float(item.description.split("=")[-1])
            seqs.append(str(item.seq))
            labels.append(label)
        return seqs, labels
