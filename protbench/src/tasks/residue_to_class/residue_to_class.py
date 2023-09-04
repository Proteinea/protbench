import abc
from typing import Dict, List, Tuple


class ResidueToClass:
    def __init__(
        self,
        label_ignore_value: int = -100,
    ):
        """A generic class for any task where the goal is to predict a class for each
            residue in a protein sequence.

        Args:
            seqs_file (str): the path to the fasta file containing the protein sequences.
            labels_file (str): the path to the fasta file containing the labels for each sequence.
                The file must have the following format:
                    >seq_id SET=train/val MASK=11100011
                    labels

                The 'SET' field determines if the corresponding sequence is part of the training or validation set.
                The 'MASK' field determines which residues should be ignored (excluded from loss and metrics computation) during training.

                Note: the 'MASK' field does not perform any attention masking on the input sequence. It only affects the loss and metrics computation.
                Note: The sequence, mask, and labels length must be the same for each sequence in the file.
            label_ignore_value (int, optional): the value of label to be ignored by loss and metrics computation.
                Defaults to -100.
        """
        super(ResidueToClass, self).__init__(label_ignore_value=label_ignore_value)
        self._data = []
        self.num_classes: int = 0
        self.class_to_id: Dict[str, int] = {}
        self.id_to_class: Dict[int, str] = {}

    @property
    def data(self) -> List[Dict[str, str | List[int]]]:
        return self._data

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

    @abc.abstractmethod
    def load_and_preprocess_data(
        self, *args, **kwargs
    ) -> Tuple[List[str], List[List[int]]]:
        """Load and preprocess the data from the given files."""
        raise NotImplementedError

    def mask_labels(self, label: List[int], mask: List[bool] | None) -> List[int]:
        """Mask the labels with the given mask by setting the masked classes to the default
            pytorch ignore index.

        Example:
            label = [0, 1, 2, 2, 1, 0]
            mask =  [1, 1, 0, 0, 1, 1]
            masked_label = [0, 1, -100, -100, 1, 0]

        Args:
            label (List[int]): encoded label
            mask (List[bool]): boolean mask with False indicating the positions to be masked (huggingface style)

        Returns:
            List[int]: masked label
        """
        if not mask:
            return label
        for i, mask_value in enumerate(mask):
            if mask_value == 0:
                label[i] = self.label_ignore_value
        return label

    def validate_lengths(
        self, seq: str, label: List[int], mask: List[bool] | None
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
