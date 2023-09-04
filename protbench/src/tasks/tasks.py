
from typing import Tuple, List, Dict

from Bio import SeqIO

from protbench.src.tasks import Task, TaskRegistry, TaskDescription


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
    def description(self) -> TaskDescription:
        return self._task_description

    def _create_description(self) -> TaskDescription:
        classification_type = "binary" if self.num_classes == 2 else "multiclass"
        return TaskDescription(
            name="sequence_to_class",
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
    def description(self) -> TaskDescription:
        return self._task_description

    def _create_description(self) -> TaskDescription:
        return TaskDescription(
            name="sequence_to_value",
            task_type=("sequence", "regression"),
            description="Generic task of predicting a value for each sequence.",
        )
