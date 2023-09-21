import pickle
from typing import Callable, Dict, List, Optional, Tuple

from protbench.tasks.residue_to_class import ResidueToClass


class PickleResidueToClass(ResidueToClass):
    def __init__(
        self,
        dataset_path: str,
        seqs_col: str,
        labels_col: str,
        class_to_id: Optional[Dict[str, int]] = None,
        mask_col: Optional[str] = None,
        preprocessing_function: Optional[Callable] = None,
        label_ignore_value: int = -100,
        validate_lengths: Optional[bool] = False,
        encode_labels: Optional[bool] = False,
        mask_labels: Optional[bool] = False,
    ):
        """
        A generic class for any task where the goal is to predict a class for each
            residue in a protein sequence. The data is loaded from a huggingface dataset.

        Args:
            dataset_url (str): the url of the huggingface dataset.
            data_files (str): the name of the data files in the dataset.
            data_key (str): the key of the data in the DatasetDict.
            seqs_col (str): the name of the column containing the sequences.
            labels_col (str): the name of the column containing the labels.
            mask_col (Optional[str], optional): the name of the column containing the masks. Defaults to None.
            preprocessing_function (Optional[Callable[[str, str, Optional[str]], Tuple[str, str, Union[str, None]]]], optional):
                a function to preprocess the a row of (seq, label, mask). Defaults to None.
            label_ignore_value (int, optional): the value of label in masked positions to be ignored by loss and metrics computation.

        """
        super(PickleResidueToClass, self).__init__(
            label_ignore_value=label_ignore_value, class_to_id=class_to_id
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
        self._check_number_of_classes()

    @property
    def data(self) -> Tuple[List[str], List[List[int]]]:
        return self._data

    def _load_and_preprocess_data(
        self,
        dataset_path: str,
        seqs_col: str,
        labels_col: str,
        mask_col: Optional[str] = None,
        preprocessing_function: Optional[Callable] = None,
        validate_lengths: Optional[bool] = False,
        encode_labels: Optional[bool] = False,
        mask_labels: Optional[bool] = False,
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
