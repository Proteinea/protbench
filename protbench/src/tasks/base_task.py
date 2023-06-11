import abc
from typing import List, Dict, Any

from protbench.src.tasks import TaskDescription


class Task(abc.ABC):
    """Base class for all tasks. All tasks must inherit from this class."""

    @abc.abstractmethod
    def get_train_data(self) -> List[Dict[str, str | Any]]:
        """Abstract method for returning the training data.

        Returns:
            List[Dict]: each element in the list is a dictionary containing the data for one sample
            with the following keys expected to be present:
                - "sequence": (str) the protein sequence
                - "label": (Any) the label of the sample
        """
        pass

    def get_val_data(self) -> List[Dict[str, str | Any]]:
        """Same as get_train_data but for validation data (if available)."""
        return []

    @property
    @abc.abstractmethod
    def task_description(self) -> TaskDescription:
        """Abstract property for returning the task description."""
        pass