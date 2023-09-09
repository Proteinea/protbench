from typing import Optional, List, Dict, Union, Callable, Tuple

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
        mask_col: Optional[str] = None,
        preprocessing_function: Optional[Callable] = None,
        label_ignore_value: int = -100,
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
        super(HuggingFaceResidueToClass, self).__init__(
            label_ignore_value=label_ignore_value
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
