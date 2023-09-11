import abc

from typing import Dict, List, Tuple, Optional


class SequenceToClass(abc.ABC):
    def __init__(
        self,
        class_to_id: Optional[Dict[str, int]] = None,
    ) -> None:
        """Generic task of predicting a class for a sequence.

        Args:
            data_file (str): path to the fasta file containing the sequences and labels.
                The file must have the following format:
                >seq_id LABEL=class
                sequence
            where SET is either train or val and LABEL is the class label.
        """
        super().__init__()

        if class_to_id:
            self.class_to_id = class_to_id
            self.id_to_class = {v: k for k, v in class_to_id.items()}
            self.num_classes = len(class_to_id)
        else:
            self.num_classes = 0
            self.class_to_id: Dict[str, int] = {}
            self.id_to_class: Dict[int, str] = {}

    @property
    @abc.abstractmethod
    def data(self) -> Tuple[List[str], List[int]]:
        raise NotImplementedError

    def encode_label(self, label: str) -> int:
        if label not in self.class_to_id:
            self.class_to_id[label] = self.num_classes
            self.id_to_class[self.num_classes] = label
            self.num_classes += 1
        return self.class_to_id[label]

    def _check_number_of_classes(self) -> None:
        if self.num_classes < 2:
            raise ValueError(
                f"Number of classes must be at least 2 but got {self.num_classes}."
            )
