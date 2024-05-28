import abc
from typing import List
from typing import Tuple

from protbench.tasks.task import Task


class SequenceToValue(Task):
    def __init__(self) -> None:
        """Generic task of predicting a value for a sequence."""
        super().__init__()

    @property
    @abc.abstractmethod
    def data(self) -> Tuple[List[str], List[float]]:
        raise NotImplementedError
