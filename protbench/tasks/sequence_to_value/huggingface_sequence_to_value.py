from __future__ import annotations

from os import PathLike
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

from datasets import load_dataset

from protbench.tasks.sequence_to_value.sequence_to_value import SequenceToValue


class HuggingFaceSequenceToValue(SequenceToValue):
    def __init__(
        self,
        dataset_url: PathLike,
        data_files: str,
        data_key: str,
        seqs_col: str,
        labels_col: str,
        preprocessing_function: Callable | None = None,
    ) -> None:
        """Generic task of predicting a class for a sequence.

        Args:
            data_file (PathLike): Path to the fasta file containing the
                sequences and labels. The file must have the following format:
                >seq_id LABEL=class
                sequence
            where SET is either train or val and LABEL is the class label.
        """
        super(HuggingFaceSequenceToValue, self).__init__()

        self._data = self.load_and_preprocess_data(
            dataset_url,
            data_files,
            data_key,
            seqs_col,
            labels_col,
            preprocessing_function,
        )

    @property
    def data(self) -> List[Dict[str, Union[str, List[int]]]]:
        return self._data

    def load_and_preprocess_data(
        self,
        dataset_url: str,
        data_files: str,
        data_key: str,
        seqs_col: str,
        labels_col: str,
        preprocessing_function: Callable | None = None,
    ) -> Tuple[List[str], List[int]]:
        # load the examples from the dataset
        dataset = load_dataset(dataset_url, data_files=data_files)
        seqs = dataset[data_key][seqs_col]
        labels = dataset[data_key][labels_col]
        for i, (seq, label) in enumerate(zip(seqs, labels)):
            if preprocessing_function is not None:
                seq, label = preprocessing_function(seq, label)
            labels[i] = label
        return seqs, labels
