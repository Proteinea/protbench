from datasets import load_dataset

from protbench.src.tasks.residue_to_class import ResidueToClass


class HuggingFaceResidueToClass(ResidueToClass):
    def __init__(
        self,
        dataset_url: str,
        data_files,
        seqs_col: str,
        labels_col: str,
        mask_col: str | None = None,
        preprocessing_function=None,
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
        super(HuggingFaceResidueToClass, self).__init__(
            label_ignore_value=label_ignore_value
        )

        self.preprocessing_function = preprocessing_function
        self._data = self.load_and_preprocess_data(
            dataset_url, data_files, seqs_col, labels_col, mask_col
        )
        self._check_number_of_classes()

    def load_and_preprocess_data(
        self,
        dataset_url: str,
        data_files,
        seqs_col: str,
        labels_col: str,
        mask_col: str | None = None,
    ) -> None:
        """Load and preprocess the data from the given files.

        Args:
            seqs_file (str): sequences file path
            labels_file (str): labels file path
        """
        dataset = load_dataset(dataset_url, data_files=data_files)
        seqs = dataset["train"][seqs_col]
        labels = dataset["train"][labels_col]
        if mask_col is not None:
            masks = dataset["train"][mask_col]
        else:
            masks = None

        for i, (seq, label) in enumerate(zip(seqs, labels)):
            if masks:
                mask = masks[i]
                seq, label, mask = self.preprocessing_function(seq, label, mask)
                self.validate_lengths(seq, label, mask)
                label = self.encode_label(label)
                label = self.mask_labels(label, mask)
            else:
                seq, label, _ = self.preprocessing_function(seq, label, None)
                label = self.encode_label(label)
                self.validate_lengths(seq, label, None)
            labels[i] = label
        return seqs, labels
