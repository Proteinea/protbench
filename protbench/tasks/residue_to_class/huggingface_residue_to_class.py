from typing import Callable, Dict, List, Optional, Tuple

from datasets import load_dataset
from protbench.tasks.residue_to_class import ResidueToClass


class HuggingFaceResidueToClass(ResidueToClass):
    def __init__(
        self,
        dataset_url: str,
        data_files: str,
        data_key: str,
        seqs_col: str,
        labels_col: str,
        class_to_id: Optional[Dict[str, int]] = None,
        mask_col: Optional[str] = None,
        preprocessing_function: Optional[Callable] = None,
        ignore_index: int = -100,
    ):
        """A generic class for any task where the goal is to predict a class
           for each residue in a protein sequence. The data is loaded from
           a huggingface dataset.

        Args:
            dataset_url (str): URL of the huggingface dataset.
            data_files (str): Name of the data files in the dataset.
            data_key (str): Key of the data in the DatasetDict.
            seqs_col (str): Name of the column containing the sequences.
            labels_col (str): Name of the column containing the labels.
            mask_col (Optional[str], optional): Name of the column
                                                containing the masks.
                                                Defaults to None.
            preprocessing_function (Optional[Callable]): Function to
                                                         preprocess the a row
                                                         of (seq, label, mask).
                                                         Defaults to None.
            ignore_index (int, optional): Value of label in masked positions to
                                          be ignored by loss and metrics
                                          computation. Defaults to -100.

        """
        super(HuggingFaceResidueToClass, self).__init__(
            ignore_index=ignore_index, class_to_id=class_to_id
        )

        self._data = self._load_and_preprocess_data(
            dataset_url,
            data_files,
            data_key,
            seqs_col,
            labels_col,
            mask_col,
            preprocessing_function,
        )
        self._check_number_of_classes()

    @property
    def data(self) -> Tuple[List[str], List[List[int]]]:
        return self._data

    def _load_and_preprocess_data(
        self,
        dataset_url: str,
        data_files: str,
        data_key: str,
        seqs_col: str,
        labels_col: str,
        mask_col: Optional[str] = None,
        preprocessing_function: Optional[Callable] = None,
    ) -> Tuple[List[str], List[List[int]]]:
        # load the examples from the dataset
        dataset = load_dataset(dataset_url, data_files=data_files)
        seqs = dataset[data_key][seqs_col]
        labels = dataset[data_key][labels_col]
        if mask_col is not None:
            masks = dataset[data_key][mask_col]
        else:
            masks = None

        for i, (seq, label) in enumerate(zip(seqs, labels)):
            if masks:
                mask = masks[i]
                seq, label, mask = preprocessing_function(seq, label, mask)
                self.validate_lengths(seq, label, mask)
                label = self.encode_label(label)
                label = self.mask_labels(label, mask)
            else:
                seq, label, _ = preprocessing_function(seq, label, None)
                label = self.encode_label(label)
                self.validate_lengths(seq, label, None)
            labels[i] = label
        return seqs, labels
