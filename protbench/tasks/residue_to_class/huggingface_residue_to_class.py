from functools import cached_property
from typing import Callable, Dict, List, Optional, Tuple

from datasets import load_dataset

from protbench.tasks.residue_to_class import ResidueToClass


class HuggingFaceResidueToClass(ResidueToClass):
    def __init__(
        self,
        dataset_url: str,
        data_files: str,
        data_key: str,
        sequences_key: str,
        labels_key: str,
        masks_key: Optional[str] = None,
        class_to_id: Optional[Dict[str, int]] = None,
        preprocessing_function: Optional[Callable] = None,
        label_ignore_value: int = -100,
        validate_length_matching: bool = True,
        encode_labels: bool = True,
        mask_labels: bool = True,
    ):
        """
        Generic class for tasks that predict a class for each residue in
        a protein sequence. Data is loaded from Huggingface datasets.

        Args:
            dataset_url (str): URL of the Huggingface dataset.
            data_files (str): Name of the data files in the dataset.
            data_key (str): Key of the data in the DatasetDict.
            seqs_col (str): Name of the column containing the sequences.
            labels_col (str): Name of the column containing the labels.
            mask_key (Optional[str], optional): Name of the column containing
                                                the masks. Defaults to None.
            preprocessing_function (Optional[Callable[[str, str, Optional[str]],
                                            Tuple[str, str, Union[str, None]]]],
                                            optional):
                Function to preprocess the a row of (seq, label, mask).
                Defaults to None.
            label_ignore_value (int, optional): Value of label in masked
                                                positions to be ignored by loss
                                                and metrics computation.
        """
        super(HuggingFaceResidueToClass, self).__init__(
            label_ignore_value=label_ignore_value, class_to_id=class_to_id
        )

        # self._data = self._load_and_preprocess_data(
        #     dataset_url,
        #     data_files,
        #     data_key,
        #     seqs_col,
        #     labels_col,
        #     mask_key,
        #     preprocessing_function,
        #     validate_lengths,
        #     encode_labels,
        #     mask_labels,
        # )
        # self._check_number_of_classes()
        self.dataset_url = dataset_url
        self.data_files = data_files
        self.data_key = data_key
        self.sequences_key = sequences_key
        self.labels_key = labels_key
        self.masks_key = masks_key
        self.preprocessing_function = preprocessing_function
        self.validate_length_matching = validate_length_matching
        self.encode_labels = encode_labels
        self._mask_labels = mask_labels

    @cached_property
    def data(self) -> Tuple[List[str], List[List[int]]]:
        results = self._load_and_preprocess_data()
        self._check_number_of_classes()
        return results

    def _load_and_preprocess_data(self) -> Tuple[List[str], List[List[int]]]:
        # load the examples from the dataset.
        dataset = load_dataset(self.dataset_url, data_files=self.data_files)
        seqs = dataset[self.data_key][self.sequences_key]
        labels = dataset[self.data_key][self.labels_key]

        if self.masks_key is not None:
            masks = dataset[self.data_key][self.masks_key]
        else:
            masks = None

        for i, (sequence, label) in enumerate(zip(seqs, labels)):
            if masks:
                mask = masks[i]

                if self.preprocessing_function is not None:
                    sequence, label, mask = self.preprocessing_function(
                        sequence, label, mask
                    )
                if self.validate_length_matching:
                    self.validate_lengths(sequence, label, mask)
                if self.encode_labels:
                    label = self.encode_label(label)
                if self._mask_labels:
                    label = self.mask_labels(label, mask)
            else:
                if self.preprocessing_function is not None:
                    sequence, label, _ = self.preprocessing_function(
                        sequence, label, None
                    )
                if self.validate_length_matching:
                    self.validate_lengths(sequence, label, None)
            labels[i] = label
        return seqs, labels
