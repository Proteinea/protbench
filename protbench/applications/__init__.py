from functools import partial
from typing import Callable
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple

from peft import TaskType

from protbench.applications import models
from protbench.applications.benchmarking_task import BenchmarkingTask
from protbench.applications.deeploc import DeepLoc
from protbench.applications.fluorescence import Fluorescence
from protbench.applications.gb1_sampled import GB1Sampled
from protbench.applications.remote_homology import RemoteHomology
from protbench.applications.solubility import Solubility
from protbench.applications.ssp3 import SSP3
from protbench.applications.ssp8 import SSP8
from protbench.applications.thermostability import Thermostability


def get_tasks(
    tasks_to_run: Optional[List] = None,
    from_embeddings: bool = False,
    tokenizer: Optional[Callable] = None,
) -> Generator[Tuple[str, BenchmarkingTask]]:
    tasks = {
        "ssp3_casp12": SSP3,
        "ssp3_casp14": SSP3,
        "ssp3_cb513": SSP3,
        "ssp3_ts115": SSP3,
        "ssp8_casp12": SSP8,
        "ssp8_casp14": SSP8,
        "ssp8_cb513": SSP8,
        "ssp8_ts115": SSP8,
        "deeploc": DeepLoc,
        "solubility": Solubility,
        "remote_homology": RemoteHomology,
        "fluorescence": Fluorescence,
        "gb1": GB1Sampled,
        "thermostability": Thermostability,
    }

    for task_name, task_cls in tasks.items():
        if task_name not in tasks_to_run:
            raise ValueError(
                f"Task {task_name} is not supported, "
                f"supported tasks are {list(tasks.keys())}."
            )

        task_instance = task_cls(
            dataset=task_name,
            from_embeddings=from_embeddings,
            tokenizer=tokenizer,
        )
        yield task_name, task_instance
