from protbench.src.tasks import Task
from datasets import load_dataset
from collections import OrderedDict
from protbench.src import utils


@utils.mark_experimental()
class SecondaryStructurePrediction(Task):
    def __init__(self, num_states, dataset="training", preprocess=False):
        if num_states not in {3, 8}:
            raise ValueError(
                "`num_states` must be equal to 3 or 8. "
                f"Recieved {num_states}."
            )

        self.supported_datasets = {
            "CASP12": "CASP12.csv",
            "CASP13": "CASP13.csv",
            "CASP14": "CASP14.csv",
            "CB513": "CB513.csv",
            "TS115": "TS115.csv",
            "training": "training_hhblits.csv",
        }

        if dataset not in self.supported_datasets:
            raise ValueError(
                "`dataset` must be one of the supported datasets "
                f"{list(self.supported_datasets.keys())}. "
                f"Recieved: {self.dataset}"
            )

        self.dataset = dataset
        self.num_states = num_states
        self.preprocess = preprocess

        self.ds = load_dataset(
            "proteinea/secondary_structure_prediction",
            data_files={self.dataset: self.supported_datasets[self.dataset]},
        )

    def load_three_states_class_mappings(self):
        class_to_id = OrderedDict([("H", 0), ("C", 1), ("E", 2)])
        id_to_class = OrderedDict([(0, "H"), (1, "C"), (2, "E")])
        return class_to_id, id_to_class

    def load_eight_states_class_mappings(self):
        class_to_id = OrderedDict(
            [
                ("S", 0),
                ("E", 1),
                ("T", 2),
                ("C", 3),
                ("I", 4),
                ("G", 5),
                ("H", 6),
                ("B", 7),
            ]
        )
        id_to_class = OrderedDict(
            [
                (0, "S"),
                (1, "E"),
                (2, "T"),
                (3, "C"),
                (4, "I"),
                (5, "G"),
                (6, "H"),
                (7, "B"),
            ]
        )
        return class_to_id, id_to_class

    def load_sequences(self):
        return self.ds[self.dataset]["input"]

    def load_labels(self):
        return self.ds[self.dataset][f"dssp{self.num_states}"]

    def load_preprocessed_labels(self, ignore_class=-100):
        labels = self.load_labels()
        masks = self.load_disorder()
        labels = self.encode_tags(labels)
        labels = self.mask_disorder(labels, masks, ignore_class=ignore_class)
        return labels

    def load_disorder(self):
        disorder = [
            list(
                map(
                    lambda x: float(x),
                    current_disorder.replace("[", "")
                    .replace("]", "")
                    .replace(",", "")
                    .split(),
                )
            )
            for current_disorder in self.ds[self.dataset]["disorder"]
        ]
        return disorder

    def mask_disorder(self, labels, masks, ignore_class=-100):
        for label, mask in zip(labels, masks):
            for i, disorder in enumerate(mask):
                if disorder == "0.0":
                    label[i] = ignore_class
        return labels

    def load_labels_mappings(self):
        if self.num_states == 3:
            return self.load_three_states_class_mappings()
        else:
            return self.load_eight_states_class_mappings()

    def encode_tags(self, labels):
        class_to_id, _ = self.load_labels_mappings(labels)
        labels = [[class_to_id[tag] for tag in doc] for doc in labels]
        return labels
