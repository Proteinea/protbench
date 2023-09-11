from typing import Dict, List, Tuple, Union

from Bio import SeqIO

from protbench.tasks.sequence_to_class import SequenceToClass


class FastaSequenceToClass(SequenceToClass):
    def __init__(
        self,
        data_file: str,
    ) -> None:
        """Generic task of predicting a class for a sequence.

        Args:
            data_file (str): Path to the fasta file containing the sequences
                             and labels.
                The file must have the following format:
                >seq_id LABEL=class
                sequence
            where SET is either train or val and LABEL is the class label.
        """
        super(FastaSequenceToClass, self).__init__()

        self._data = self.load_and_preprocess_data(data_file)
        self._check_number_of_classes()

    @property
    def data(self) -> List[Dict[str, Union[str, List[int]]]]:
        return self._data

    def load_and_preprocess_data(
        self, data_file
    ) -> Tuple[List[str], List[int]]:
        sequences, labels = [], []
        for item in SeqIO.parse(data_file, "fasta"):
            label = item.description.split("=")[-1]
            label = self.encode_label(label)
            sequences.append(item.seq)
            labels.append(label)
        return sequences, labels
