from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from datasets import load_dataset

from protbench.tasks.sequence_to_class.sequence_to_class import SequenceToClass


class HuggingFaceSequenceToClass(SequenceToClass):
    def __init__(
        self,
        dataset_url: str,
        data_files: str,
        data_key: str,
        seqs_col: str,
        labels_col: str,
        class_to_id: Optional[Dict[str, int]] = None,
        preprocessing_function: Optional[Callable] = None,
    ) -> None:
        """A generic class for any task where the goal is to predict a class
           for each residue in a protein sequence. The data is loaded from
           a huggingface dataset.

        Args:
            dataset_url (str): URL of the huggingface dataset.
            data_files (str): Name of the data files in the dataset.
            data_key (str): Key of the data in the DatasetDict.
            seqs_col (str): Name of the column containing the sequences.
            labels_col (str): Name of the column containing the labels.
            class_to_id (Optional[Dict[str, int]]): Dictionary containing class
                                                    names and their
                                                    corresponding ids.
            preprocessing_function (Optional[Callable]): Function to
                                                         preprocess the a row
                                                         of (seq, label, mask).
                                                         Defaults to None.
        """
        super(HuggingFaceSequenceToClass, self).__init__(
            class_to_id=class_to_id
        )

        self._data = self.load_and_preprocess_data(
            dataset_url,
            data_files,
            data_key,
            seqs_col,
            labels_col,
            preprocessing_function,
        )
        self._check_number_of_classes()

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
        preprocessing_function: Optional[Callable] = None,
    ) -> Tuple[List[str], List[int]]:
        # load the examples from the dataset
        dataset = load_dataset(dataset_url, data_files=data_files)
        seqs = dataset[data_key][seqs_col]
        labels = dataset[data_key][labels_col]
        for i, (seq, label) in enumerate(zip(seqs, labels)):
            if preprocessing_function is not None:
                seq, label = preprocessing_function(seq, label)
            label = self.encode_label(label)
            labels[i] = label
        return seqs, labels
