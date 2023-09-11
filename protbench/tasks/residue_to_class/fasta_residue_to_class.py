from Bio import SeqIO


from typing import List, Tuple, Union, Dict

from protbench.tasks.residue_to_class import ResidueToClass


class FastaResidueToClass(ResidueToClass):
    def __init__(
        self,
        seqs_file: str,
        labels_file: str,
        label_ignore_value: int = -100,
    ):
        """A generic class for any task where the goal is to predict a class for each
            residue in a protein sequence.

        Args:
            seqs_file (str): the path to the fasta file containing the protein sequences.
            labels_file (str): the path to the fasta file containing the labels for each sequence.
                The file must have the following format:
                    >seq_id MASK=11100011
                    labels

                The 'SET' field determines if the corresponding sequence is part of the training or validation set.
                The 'MASK' field determines which residues should be ignored (excluded from loss and metrics computation) during training.

                Note: the 'MASK' field does not perform any attention masking on the input sequence. It only affects the loss and metrics computation.
                Note: The sequence, mask, and labels length must be the same for each sequence in the file.
            label_ignore_value (int, optional): the value of label to be ignored by loss and metrics computation.
                Defaults to -100.
        """
        super(FastaResidueToClass, self).__init__(label_ignore_value=label_ignore_value)

        self._data = self.load_and_preprocess_data(seqs_file, labels_file)
        self._check_number_of_classes()

    @property
    def data(self) -> Tuple[List[str], List[List[int]]]:
        return self._data

    def load_and_preprocess_data(
        self, seqs_file: str, labels_file: str
    ) -> Tuple[List[str], List[List[int]]]:
        """Load and preprocess the data from the given files.

        Args:
            seqs_file (str): sequences file path
            labels_file (str): labels file path
        """
        seqs = {item.id: str(item.seq) for item in SeqIO.parse(seqs_file, "fasta")}
        inputs = []
        labels = []
        for item in SeqIO.parse(labels_file, "fasta"):
            if item.id not in seqs:
                raise KeyError(
                    f"Sequence with id {item.id} in {labels_file} not found in {seqs_file}."
                )
            mask = self._get_mask_from_desc_if_available(item.description)
            label = self.encode_label(str(item.seq))
            self.validate_lengths(seqs[item.id], label, mask)
            label = self.mask_labels(label, mask)
            inputs.append(seqs[item.id])
            labels.append(label)

        return inputs, labels

    def _get_mask_from_desc_if_available(
        self, label_description: str
    ) -> Union[List[bool], None]:
        """Parse the label description to extract the sequence set and mask.

        Args:
            label_description (str): the label description string.

        Returns:
            Tuple[str, List[bool]]: the sequence set and the mask.
        """
        label_description_split = label_description.split(" ")
        if len(label_description_split) == 1:
            return None

        mask = label_description_split[1]
        if not mask.startswith("MASK="):
            raise ValueError(
                f"Expected label description to start with 'MASK=' but got {mask}."
            )

        mask = [bool(int(value)) for value in mask.split("=")[-1]]
        return mask
