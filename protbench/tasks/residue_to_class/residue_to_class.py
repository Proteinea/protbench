import abc
from typing import Dict, List, Optional, Tuple

from protbench.tasks.task import Task


class ResidueToClass(Task):
    def __init__(
        self,
        ignore_index: int = -100,
        class_to_id: Optional[Dict[str, int]] = None,
    ):
        """A generic class for any task where the goal is to predict a class
           for each residue in a protein sequence.

        Args:
            ignore_index (int, optional): Value of label to be ignored by loss
                                          and metrics computation.
                                          Defaults to -100.
            class_to_id (Optional[Dict[str, int]]): Dictionary containing class
                                                    names and their
                                                    corresponding ids.
        """
        self.ignore_index = ignore_index
        if class_to_id:
            self.class_to_id = class_to_id
            self.num_classes = len(class_to_id)
            self.id_to_class = {v: k for k, v in class_to_id.items()}
        else:
            self.num_classes: int = 0
            self.class_to_id: Dict[str, int] = {}
            self.id_to_class: Dict[int, str] = {}

    @property
    @abc.abstractmethod
    def data(self) -> Tuple[List[str], List[List[int]]]:
        raise NotImplementedError

    def encode_label(self, label: str) -> List[int]:
        """Encode a label string into a list of integers.

        Args:
            label (str): the label string to encode.

        Returns:
            List[int]: the encoded label.
        """
        encoded_label = [0] * len(label)
        for i, cls in enumerate(label):
            if cls not in self.class_to_id:
                self.class_to_id[cls] = self.num_classes
                self.id_to_class[self.num_classes] = cls
                self.num_classes += 1
            encoded_label[i] = self.class_to_id[cls]
        return encoded_label

    def mask_labels(
        self, label: List[int], mask: Optional[List[bool]]
    ) -> List[int]:
        """Mask the labels with the given mask by setting the masked classes to the default
            pytorch ignore index.

        Example:
            label = [0, 1, 2, 2, 1, 0]
            mask =  [1, 1, 0, 0, 1, 1]
            masked_label = [0, 1, -100, -100, 1, 0]

        Args:
            label (List[int]): Encoded label
            mask (List[bool]): Boolean mask with False indicating the positions
                               to be masked (huggingface style)

        Returns:
            List[int]: masked label
        """
        if not mask:
            return label
        for i, mask_value in enumerate(mask):
            if mask_value == 0:
                label[i] = self.ignore_index
        return label

    def validate_lengths(
        self, seq: str, label: List[int], mask: Optional[List[bool]]
    ) -> None:
        if mask:
            if len(seq) != len(label) or len(seq) != len(mask):
                raise ValueError(
                    "Sequence, label and mask must be of same length but got "
                    f"{len(seq)}, {len(label)} and {len(mask)}."
                )
        else:
            if len(seq) != len(label):
                raise ValueError(
                    "Sequence and label must be of same length but got "
                    f"{len(seq)} and {len(label)}."
                )

    def _check_number_of_classes(self) -> None:
        if self.num_classes < 2:
            raise ValueError(
                f"Number of classes must be at least 2 but got {self.num_classes}."
            )
