import logging
from typing import Dict, Optional, Callable, Type

from protbench.src.tasks import Task


class TaskRegistry:
    """Central repository for all tasks.

    - To register a new task you can use the `add_task` decorator as follows:
    @TaskRegistry.add_task('my_new_task')
    class MyNewTask(Task):
    ...

    - Or you can use the `add_task` method directly:

    class MyNewTask(Task):
    ...
    TaskRegistry.add_task('my_new_task', MyNewTask)
    """

    task_name_map: Dict[str, Type[Task]] = {}

    @classmethod
    def register(
        cls,
        task_name: str,
        task_cls: Optional[Type[Task]] = None,
    ) -> Callable | Type[Task]:
        """Register a new task. This can be used as a decorator providing only the task_name
        or directly as a method providing the task_name and task class.

        Args:
            task_name (str): name of the task. Must be unique. Same name will be used in
                the config file to refer to the task.
            task_cls (Optional[Type[Task]], optional): task class. Defaults to None.
        """
        if task_name in cls.task_name_map:
            raise ValueError(
                f"Task {task_name} already exists in the registry. "
                f"Please choose a different name."
            )
        if task_cls is None:  # expected when using decorator
            return lambda task_cls: cls.register(task_name, task_cls)
        if not issubclass(task_cls, Task):
            logging.warning(
                f"Task {task_name} does not inherit from the base Task class."
            )
        cls.task_name_map[task_name] = task_cls
        return task_cls
