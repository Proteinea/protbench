import abc
from typing import List, Dict, Any, Optional

from protbench.src.tasks import TaskDescription


class Task(abc.ABC):
    """Base class for all tasks. All tasks must inherit from this class."""

    def __init__(self, label_ignore_value: Optional[int] = None):
        """Initialize the task.

        Args:
            label_ignore_value (Optional[int], optional): The value to be ignored in
                the labels (if applicable). Defaults to None.
        """
        self.label_ignore_value = label_ignore_value

    @property
    @abc.abstractmethod
    def train_data(self) -> List[Dict[str, str | Any]]:
        """Abstract method for returning the training data.

        Returns:
            List[Dict]: each element in the list is a dictionary containing the data for one sample
            with the following keys expected to be present:
                - "sequence": (str) the protein sequence
                - "label": (Any) the label of the sample
        """
        pass

    @property
    def val_data(self) -> List[Dict[str, str | Any]]:
        """Same as get_train_data but for validation data (if available)."""
        return []

    @property
    @abc.abstractmethod
    def task_description(self) -> TaskDescription:
        """Abstract property for returning the task description."""
        pass
