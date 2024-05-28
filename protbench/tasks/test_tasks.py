import os
import random
import unittest
from typing import List
from typing import Tuple

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from protbench.tasks.residue_to_class.residue_to_class import ResidueToClass
from protbench.tasks.sequence_to_class.sequence_to_class import SequenceToClass
from protbench.tasks.sequence_to_value.sequence_to_value import SequenceToValue

random.seed(42)


class TestResidueToClass(unittest.TestCase):
    def generate_random_sequence(self, length: int) -> str:
        return "".join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=length))

    def generate_random_mask(self, length: int) -> str:
        return "".join(random.choices("01", k=length))

    def generate_random_labels(self, labels_set: List[str], length) -> str:
        return "".join(random.choices(labels_set, k=length))

    def write_data_to_fasta(
        self,
        sequences_file: str,
        labels_file: str,
        sequences: List[str],
        labels: List[str],
        masks: List[str],
        sets: List[str],
    ):
        all_sequences = []
        all_labels = []
        for i, (seq, label, mask, set) in enumerate(
            zip(sequences, labels, masks, sets)
        ):
            seq_record = SeqRecord(Seq(seq), id=f"seq_{i}", description="")
            label_record = SeqRecord(
                Seq(label), id=f"seq_{i}", description=f"SET={set} MASK={mask}"
            )
            all_sequences.append(seq_record)
            all_labels.append(label_record)

        with open(sequences_file, "w") as output_handle:
            SeqIO.write(all_sequences, output_handle, "fasta")
        with open(labels_file, "w") as output_handle:
            SeqIO.write(all_labels, output_handle, "fasta")

    def create_data_files(
        self,
        sequences_file: str,
        labels_file: str,
        labels_set: List[str],
        num_train: int,
        num_val: int,
    ) -> Tuple[List[str], List[str], List[str]]:
        seq_min_length = 80
        seq_max_length = 120

        sequences = []
        labels = []
        masks = []
        sets = []

        for _ in range(num_train):
            seq = self.generate_random_sequence(
                length=random.randint(seq_min_length, seq_max_length)
            )
            label = self.generate_random_labels(
                labels_set=labels_set, length=len(seq)
            )
            mask = self.generate_random_mask(len(seq))
            set = "train"
            sequences.append(seq)
            labels.append(label)
            masks.append(mask)
            sets.append(set)

        for _ in range(num_val):
            seq = self.generate_random_sequence(
                length=random.randint(seq_min_length, seq_max_length)
            )
            label = self.generate_random_labels(
                labels_set=labels_set, length=len(seq)
            )
            mask = self.generate_random_mask(len(seq))
            set = "val"
            sequences.append(seq)
            labels.append(label)
            masks.append(mask)
            sets.append(set)

        self.write_data_to_fasta(
            sequences_file, labels_file, sequences, labels, masks, sets
        )
        return sequences, labels, masks

    def test_residue_to_class(self):
        sequences_file = "test_sequences.fasta"
        labels_file = "test_labels.fasta"
        labels_set = ["A", "B", "C"]
        num_train = 800
        num_val = 200
        ignore_index = -1

        sequences, labels, masks = self.create_data_files(
            sequences_file, labels_file, labels_set, num_train, num_val
        )

        task = ResidueToClass(
            seqs_file=sequences_file,
            labels_file=labels_file,
            label_ignore_value=ignore_index,
        )

        # Test that all classes have been encoded properly
        self.assertEqual(len(task.class_to_id), len(labels_set))
        self.assertEqual(len(task.id_to_class), len(labels_set))
        self.assertEqual(task.num_classes, len(labels_set))
        [self.assertIn(label, task.class_to_id) for label in labels_set]
        [
            self.assertIn(task.class_to_id[label], task.id_to_class)
            for label in labels_set
        ]

        self.assertEqual(len(task.train_data), num_train)
        self.assertEqual(len(task.val_data), num_val)

        for i, (seq, label, mask) in enumerate(
            zip(sequences[:num_train], labels[:num_train], masks[:num_train])
        ):
            test_seq, test_label = (
                task.train_data[i]["sequence"],
                task.train_data[i]["label"],
            )
            self.assertEqual(test_seq, seq)
            for j, (class_label, mask_value) in enumerate(zip(label, mask)):
                if mask_value == "1":
                    self.assertEqual(
                        test_label[j], task.class_to_id[class_label]
                    )
                else:
                    self.assertEqual(test_label[j], ignore_index)

        for i, (seq, label, mask) in enumerate(
            zip(sequences[num_train:], labels[num_train:], masks[num_train:])
        ):
            test_seq, test_label = (
                task.val_data[i]["sequence"],
                task.val_data[i]["label"],
            )
            self.assertEqual(test_seq, seq)
            for j, (class_label, mask_value) in enumerate(zip(label, mask)):
                if mask_value == "1":
                    self.assertEqual(
                        test_label[j], task.class_to_id[class_label]
                    )
                else:
                    self.assertEqual(test_label[j], ignore_index)

        os.remove(sequences_file)
        os.remove(labels_file)


class TestSequenceToClass(unittest.TestCase):
    def generate_random_sequence(self, length: int) -> str:
        return "".join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=length))

    def generate_random_label(self, labels_set: List[str]) -> str:
        return random.choice(labels_set)

    def write_data_to_fasta(
        self,
        data_file: str,
        sequences: List[str],
        labels: List[str],
        sets: List[str],
    ):
        samples = []
        for i, (seq, label, set) in enumerate(zip(sequences, labels, sets)):
            sample = SeqRecord(
                Seq(seq), id=f"seq_{i}", description=f"SET={set} LABEL={label}"
            )
            samples.append(sample)

        with open(data_file, "w") as output_handle:
            SeqIO.write(samples, output_handle, "fasta")

    def create_data_files(
        self,
        data_file: str,
        labels_set: List[str],
        num_train: int,
        num_val: int,
    ) -> Tuple[List[str], List[str]]:
        seq_min_length = 80
        seq_max_length = 120

        sequences = []
        labels = []
        sets = []

        for _ in range(num_train):
            seq = self.generate_random_sequence(
                length=random.randint(seq_min_length, seq_max_length)
            )
            label = self.generate_random_label(labels_set=labels_set)
            set = "train"
            sequences.append(seq)
            labels.append(label)
            sets.append(set)

        for _ in range(num_val):
            seq = self.generate_random_sequence(
                length=random.randint(seq_min_length, seq_max_length)
            )
            label = self.generate_random_label(labels_set=labels_set)
            set = "val"
            sequences.append(seq)
            labels.append(label)
            sets.append(set)

        self.write_data_to_fasta(data_file, sequences, labels, sets)
        return sequences, labels

    def test_sequence_to_class(self):
        data_file = "test_data.fasta"
        labels_set = ["A", "B", "C"]
        num_train = 800
        num_val = 200

        sequences, labels = self.create_data_files(
            data_file,
            labels_set,
            num_train,
            num_val,
        )

        task = SequenceToClass(data_file=data_file)

        # Test that all classes have been encoded properly
        self.assertEqual(len(task.class_to_id), len(labels_set))
        self.assertEqual(len(task.id_to_class), len(labels_set))
        self.assertEqual(task.num_classes, len(labels_set))
        [self.assertIn(label, task.class_to_id) for label in labels_set]
        [
            self.assertIn(task.class_to_id[label], task.id_to_class)
            for label in labels_set
        ]

        self.assertEqual(len(task.train_data), num_train)
        self.assertEqual(len(task.val_data), num_val)

        for i, (seq, label) in enumerate(
            zip(sequences[:num_train], labels[:num_train])
        ):
            test_seq, test_label = (
                task.train_data[i]["sequence"],
                task.train_data[i]["label"],
            )
            self.assertEqual(test_seq, seq)
            self.assertEqual(test_label, task.class_to_id[label])

        for i, (seq, label) in enumerate(
            zip(sequences[num_train:], labels[num_train:])
        ):
            test_seq, test_label = (
                task.val_data[i]["sequence"],
                task.val_data[i]["label"],
            )
            self.assertEqual(test_seq, seq)
            self.assertEqual(test_label, task.class_to_id[label])

        os.remove(data_file)


class TestSequenceToValue(unittest.TestCase):
    def generate_random_sequence(self, length: int) -> str:
        return "".join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=length))

    def generate_random_target(self) -> float:
        return random.random()

    def write_data_to_fasta(
        self,
        data_file: str,
        sequences: List[str],
        targets: List[float],
        sets: List[str],
    ):
        samples = []
        for i, (seq, target, set) in enumerate(zip(sequences, targets, sets)):
            sample = SeqRecord(
                Seq(seq),
                id=f"seq_{i}",
                description=f"SET={set} VALUE={target}",
            )
            samples.append(sample)

        with open(data_file, "w") as output_handle:
            SeqIO.write(samples, output_handle, "fasta")

    def create_data_files(
        self,
        data_file: str,
        num_train: int,
        num_val: int,
    ) -> Tuple[List[str], List[float]]:
        seq_min_length = 80
        seq_max_length = 120

        sequences = []
        targets = []
        sets = []

        for _ in range(num_train):
            seq = self.generate_random_sequence(
                length=random.randint(seq_min_length, seq_max_length)
            )
            target = self.generate_random_target()
            set = "train"
            sequences.append(seq)
            targets.append(target)
            sets.append(set)

        for _ in range(num_val):
            seq = self.generate_random_sequence(
                length=random.randint(seq_min_length, seq_max_length)
            )
            target = self.generate_random_target()
            set = "val"
            sequences.append(seq)
            targets.append(target)
            sets.append(set)

        self.write_data_to_fasta(data_file, sequences, targets, sets)
        return sequences, targets

    def test_sequence_to_value(self):
        data_file = "test_data.fasta"
        num_train = 800
        num_val = 200

        sequences, targets = self.create_data_files(
            data_file,
            num_train,
            num_val,
        )

        task = SequenceToValue(data_file=data_file)

        self.assertEqual(len(task.train_data), num_train)
        self.assertEqual(len(task.val_data), num_val)

        for i, (seq, target) in enumerate(
            zip(sequences[:num_train], targets[:num_train])
        ):
            test_seq, test_target = (
                task.train_data[i]["sequence"],
                task.train_data[i]["label"],
            )
            self.assertEqual(test_seq, seq)
            self.assertEqual(test_target, target)

        for i, (seq, target) in enumerate(
            zip(sequences[num_train:], targets[num_train:])
        ):
            test_seq, test_target = (
                task.val_data[i]["sequence"],
                task.val_data[i]["label"],
            )
            self.assertEqual(test_seq, seq)
            self.assertEqual(test_target, target)

        os.remove(data_file)
