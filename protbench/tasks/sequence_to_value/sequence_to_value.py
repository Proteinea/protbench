import abc
from typing import List, Tuple

from protbench.tasks.task import Task


class SequenceToValue(Task):
    def __init__(
        self,
    ) -> None:
        """Generic task of predicting a value for a sequence.

        Args:
            data_file (str): path to the fasta file containing the sequences and labels.
                The file must have the following format:
                >seq_id VALUE=value
                sequence

        """
        super().__init__()

    @property
    @abc.abstractmethod
    def data(self) -> Tuple[List[str], List[float]]:
        raise NotImplementedError
