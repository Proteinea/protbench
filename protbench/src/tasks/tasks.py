from typing import Tuple, List, Dict

from Bio import SeqIO

from protbench.src.tasks import Task, TaskRegistry, TaskDescription


@TaskRegistry.register("residue_to_class")
class ResidueToClass(Task):
    def __init__(
        self,
        seqs_file: str,
        labels_file: str,
        ignore_index: int = -100,
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
            ignore_index (int, optional): the index to ignore in the loss computation. Defaults to -100 (default value for CrossEntropyLoss ignore index).
        """
        super(ResidueToClass, self).__init__()

        self._train_data: List[Dict[str, str | List[int]]] = []
        self._val_data: List[Dict[str, str | List[int]]] = []

        self.num_classes: int = 0
        self.class_to_id: Dict[str, int] = {}
        self.id_to_class: Dict[int, str] = {}
        self.ignore_index = ignore_index

        self.load_and_preprocess_data(seqs_file, labels_file)
        self._check_number_of_classes()

        self._task_description = self._create_description()

    @property
    def train_data(self) -> List[Dict[str, str | List[int]]]:
        return self._train_data

    @property
    def val_data(self) -> List[Dict[str, str | List[int]]]:
        return self._val_data

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

    def load_and_preprocess_data(self, seqs_file: str, labels_file: str) -> None:
        """Load and preprocess the data from the given files.

        Args:
            seqs_file (str): sequences file path
            labels_file (str): labels file path
        """
        seqs = {item.id: str(item.seq) for item in SeqIO.parse(seqs_file, "fasta")}
        train_data = []
        val_data = []

        for item in SeqIO.parse(labels_file, "fasta"):
            if item.id not in seqs:
                raise KeyError(
                    f"Sequence with id {item.id} in {labels_file} not found in {seqs_file}."
                )
            seq_set, mask = self._parse_label_description(item.description)
            label = self.encode_label(str(item.seq))
            self.validate_lengths(seqs[item.id], label, mask)
            label = self.mask_labels(label, mask)

            example = {"sequence": seqs[item.id], "label": label}
            if seq_set == "train":
                train_data.append(example)
            else:
                val_data.append(example)

        self._train_data = train_data
        self._val_data = val_data

    def mask_labels(self, label: List[int], mask: List[bool]) -> List[int]:
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
        for i, mask_value in enumerate(mask):
            if mask_value == 0:
                label[i] = self.ignore_index
        return label

    def _parse_label_description(
        self, label_description: str
    ) -> Tuple[str, List[bool]]:
        """Parse the label description to extract the sequence set and mask.

        Args:
            label_description (str): the label description string.

        Returns:
            Tuple[str, List[bool]]: the sequence set and the mask.
        """
        label_description_split = label_description.split(" ")
        seq_set, mask = label_description_split[1], label_description_split[2]
        seq_set = seq_set.split("=")[-1].lower()
        if seq_set not in {"train", "val"}:
            raise ValueError(
                f"Sequences can only belong to 'train' or 'val' sets but got {seq_set}."
            )
        mask = [bool(int(value)) for value in mask.split("=")[-1]]
        return seq_set, mask

    def validate_lengths(self, seq: str, label: List[int], mask: List[bool]) -> None:
        if len(seq) != len(label) or len(seq) != len(mask):
            raise ValueError(
                "Sequence, label and mask must be of same length but got "
                f"{len(seq)}, {len(label)} and {len(mask)}."
            )

    @property
    def task_description(self) -> TaskDescription:
        return self._task_description

    def _create_description(self) -> TaskDescription:
        classification_type = "binary" if self.num_classes == 2 else "multiclass"
        return TaskDescription(
            name="residue_to_class",
            task_type=("token", f"{classification_type}_classification"),
            description="Generic task of predicting a class for each residue in sequence.",
        )

    def _check_number_of_classes(self) -> None:
        if self.num_classes < 2:
            raise ValueError(
                f"Number of classes must be at least 2 but got {self.num_classes}."
            )


@TaskRegistry.register("sequence_to_class")
class SequenceToClass(Task):
    def __init__(self, data_file: str) -> None:
        """Generic task of predicting a class for a sequence.

        Args:
            data_file (str): path to the fasta file containing the sequences and labels.
                The file must have the following format:
                >seq_id SET=train/val LABEL=class
                sequence
            where SET is either train or val and LABEL is the class label.
        """
        super().__init__()
        self._train_data: List[Dict[str, str | int]] = []
        self._val_data: List[Dict[str, str | int]] = []

        self.num_classes: int = 0
        self.class_to_id: Dict[str, int] = {}
        self.id_to_class: Dict[int, str] = {}

        self.load_and_preprocess_data(data_file)
        self._check_number_of_classes()

        self._task_description = self._create_description()

    @property
    def train_data(self) -> List[Dict[str, str | int]]:
        return self._train_data

    @property
    def val_data(self) -> List[Dict[str, str | int]]:
        return self._val_data

    def encode_label(self, label: str) -> int:
        if label not in self.class_to_id:
            self.class_to_id[label] = self.num_classes
            self.id_to_class[self.num_classes] = label
            self.num_classes += 1
        return self.class_to_id[label]

    def load_and_preprocess_data(self, data_file) -> None:
        train_data = []
        val_data = []

        for item in SeqIO.parse(data_file, "fasta"):
            seq_set, label = self._parse_sequence_description(item.description)
            label = self.encode_label(label)
            if seq_set == "train":
                train_data.append({"sequence": str(item.seq), "label": label})
            else:
                val_data.append({"sequence": str(item.seq), "label": label})
        self._train_data = train_data
        self._val_data = val_data

    def _parse_sequence_description(self, sequence_description: str) -> Tuple[str, str]:
        sequence_description_split = sequence_description.split(" ")
        seq_set, label = sequence_description_split[1], sequence_description_split[2]
        seq_set = seq_set.split("=")[-1].lower()
        if seq_set not in {"train", "val"}:
            raise ValueError(
                f"Sequences can only belong to 'train' or 'val' sets but got {seq_set}."
            )
        label = label.split("=")[-1]
        return seq_set, label

    @property
    def task_description(self) -> TaskDescription:
        return self._task_description

    def _create_description(self) -> TaskDescription:
        classification_type = "binary" if self.num_classes == 2 else "multiclass"
        return TaskDescription(
            name="residue_to_class",
            task_type=("sequence", f"{classification_type}_classification"),
            description="Generic task of predicting a class for each sequence.",
        )

    def _check_number_of_classes(self) -> None:
        if self.num_classes < 2:
            raise ValueError(
                f"Number of classes must be at least 2 but got {self.num_classes}."
            )


@TaskRegistry.register("sequence_to_value")
class SequenceToValue(Task):
    def __init__(self, data_file: str) -> None:
        """Generic task of predicting a value for a sequence.

        Args:
            data_file (str): path to the fasta file containing the sequences and labels.
                The file must have the following format:
                >seq_id SET=train/val VALUE=value
                sequence

            where SET is either train or val and VALUE is the value to predict.
        """
        super().__init__()
        self._train_data: List[Dict[str, str | float]] = []
        self._val_data: List[Dict[str, str | float]] = []

        self.load_and_preprocess_data(data_file)

        self._task_description = self._create_description()

    @property
    def train_data(self) -> List[Dict[str, str | float]]:
        return self._train_data

    @property
    def val_data(self) -> List[Dict[str, str | float]]:
        return self._val_data

    def load_and_preprocess_data(self, data_file) -> None:
        train_data = []
        val_data = []

        for item in SeqIO.parse(data_file, "fasta"):
            seq_set, label = self._parse_sequence_description(item.description)
            if seq_set == "train":
                train_data.append({"sequence": str(item.seq), "label": label})
            else:
                val_data.append({"sequence": str(item.seq), "label": label})
        self._train_data = train_data
        self._val_data = val_data

    def _parse_sequence_description(
        self, sequence_description: str
    ) -> Tuple[str, float]:
        sequence_description_split = sequence_description.split(" ")
        seq_set, label = sequence_description_split[1], sequence_description_split[2]
        seq_set = seq_set.split("=")[-1].lower()
        if seq_set not in {"train", "val"}:
            raise ValueError(
                f"Sequences can only belong to 'train' or 'val' sets but got {seq_set}."
            )
        label = float(label.split("=")[-1])
        return seq_set, label

    @property
    def task_description(self) -> TaskDescription:
        return self._task_description

    def _create_description(self) -> TaskDescription:
        return TaskDescription(
            name="residue_to_class",
            task_type=("sequence", "regression"),
            description="Generic task of predicting a value for each sequence.",
        )
