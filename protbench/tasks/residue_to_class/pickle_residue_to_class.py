from __future__ import annotations

import pickle
from os import PathLike
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

from protbench.tasks.residue_to_class.residue_to_class import ResidueToClass


class PickleResidueToClass(ResidueToClass):
    def __init__(
        self,
        dataset_path: PathLike,
        seqs_col: str,
        labels_col: str,
        class_to_id: Dict[str, int] | None = None,
        mask_col: str | None = None,
        preprocessing_function: Callable | None = None,
        ignore_index: int = -100,
        validate_lengths: bool | None = False,
        encode_labels: bool | None = False,
        mask_labels: bool | None = False,
        num_classes: int | None = None,
    ):
        """A generic class for any task where the goal is to predict a class
           for each residue in a protein sequence. The data is loaded from a
           huggingface dataset.

        Args:
            dataset_url (PathLike): URL of the huggingface dataset.
            data_files (str): Name of the data files in the dataset.
            data_key (str): Key of the data in the DatasetDict.
            seqs_col (str): Name of the column containing the sequences.
            labels_col (str): Name of the column containing the labels.
            mask_col (str | None, optional): Name of the column containing
                the masks. Defaults to None.
            preprocessing_function (Callable | None): Function to
                preprocess the a row of (seq, label, mask). Defaults to None.
            ignore_index (int, optional): Value of label in masked
                positions to be ignored by loss and metrics computation.

        """
        super(PickleResidueToClass, self).__init__(
            ignore_index=ignore_index, class_to_id=class_to_id
        )

        self._data = self._load_and_preprocess_data(
            dataset_path,
            seqs_col,
            labels_col,
            mask_col,
            preprocessing_function,
            validate_lengths,
            encode_labels,
            mask_labels,
        )
        if num_classes is None:
            self._check_number_of_classes()
        else:
            self.num_classes = num_classes

    @property
    def data(self) -> Tuple[List[str], List[List[int]]]:
        return self._data

    def _load_and_preprocess_data(
        self,
        dataset_path: PathLike,
        seqs_col: str,
        labels_col: str,
        mask_col: str | None = None,
        preprocessing_function: Callable | None = None,
        validate_lengths: bool = False,
        encode_labels: bool = False,
        mask_labels: bool = False,
    ) -> Tuple[List[str], List[List[int]]]:
        # load the examples from the dataset
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)

        labels = []
        seqs = []
        for i, example in enumerate(dataset):
            seq = example[seqs_col]
            label = example[labels_col]
            if mask_col is not None:
                mask = example[mask_col]
                seq, label, mask = preprocessing_function(seq, label, mask)
                if validate_lengths:
                    self.validate_lengths(seq, label, mask)
                if encode_labels:
                    label = self.encode_label(label)
                if mask_labels:
                    label = self.mask_labels(label, mask)

            else:
                seq, label, _ = preprocessing_function(seq, label, None)

                if encode_labels:
                    label = self.encode_label(label)
                if validate_lengths:
                    self.validate_lengths(seq, label, None)
            seqs.append(seq)
            labels.append(label)
        return seqs, labels
