from typing import Literal, get_args, Optional, Tuple, Dict

from protbench.src.tasks import utils

ENTITY = Literal["token", "sequence"]
OPERATION = Literal[
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "regression",
    "generation",
]


class TaskDescription:
    def __init__(
        self,
        name: str,
        task_type: Tuple[ENTITY, OPERATION],
        description: str,
    ):
        """Creates a task description.

        Args:
            name (str): name of the task.
            task_type (Tuple[Entities, Operations]): type of the task in the form of a tuple of entity and operation.
            description (str): brief description of the task.
        """
        self.task_name = name
        self.task_type = task_type
        self.task_description = description

    @property
    def task_name(self) -> str:
        return self._task_name

    @task_name.setter
    def task_name(self, task_name: str) -> None:
        utils.validate_type("task_name", task_name, str)
        self._task_name = task_name

    @property
    def task_type(self) -> Dict[str, str]:
        return self._task_type

    @task_type.setter
    def task_type(self, task_type: Tuple[ENTITY, OPERATION]) -> None:
        entity, operation = task_type
        valid_entities = get_args(ENTITY)
        valid_operations = get_args(OPERATION)
        if entity not in valid_entities:
            raise ValueError(
                f"Expected entity to be one of {', '.join(valid_entities)}"
                f" but got '{entity}' instead."
            )
        elif operation not in valid_operations:
            raise ValueError(
                f"Expected operation to be one of {', '.join(valid_operations)}"
                f" but got '{operation}' instead."
            )
        else:
            self._task_type = {"entity": entity, "operation": operation}

    @property
    def task_description(self) -> str:
        return self.task_description

    @task_description.setter
    def task_description(self, task_description) -> None:
        utils.validate_type("task_description", task_description, str)
        self._task_description = task_description

    @property
    def num_examples(self) -> int | None:
        return self._num_examples

    @num_examples.setter
    def num_examples(self, num_examples: Optional[int]) -> None:
        if num_examples is None:
            self._num_examples = None
            return
        utils.validate_type("num_examples", num_examples, int)
        if num_examples < 1:
            raise ValueError(
                f"Expected num_examples to be greater than zero "
                "but got {num_examples_} instead."
            )
        else:
            self._num_examples = num_examples

    def __repr__(self) -> str:
        return (
            "*" * 50
            + "\n"
            + f"Task Name: {self._task_name}\n"
            + f"Task Type: {self._task_type['entity']} {self._task_type['operation']}\n"
            + f"Task Description: {self._task_description}\n"
            + "*" * 50
        )
