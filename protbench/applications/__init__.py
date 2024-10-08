from __future__ import annotations

from typing import Generator
from typing import List
from typing import Tuple

from protbench.applications import pretrained
from protbench.applications.benchmarking_task import BenchmarkingTask
from protbench.applications.deeploc import DeepLoc
from protbench.applications.fluorescence import Fluorescence
from protbench.applications.gb1_sampled import GB1Sampled
from protbench.applications.pretrained.pretrained import PretrainedModelWrapper
from protbench.applications.pretrained.pretrained import \
    initialize_model_from_checkpoint
from protbench.applications.remote_homology import RemoteHomology
from protbench.applications.solubility import Solubility
from protbench.applications.ssp3 import SSP3
from protbench.applications.ssp8 import SSP8
from protbench.applications.thermostability import Thermostability

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
        "gb1_sampled": GB1Sampled,
        "thermostability": Thermostability,
    }


def get_task(identifier) -> BenchmarkingTask:
    if identifier not in tasks:
        raise ValueError(
            f"Task {identifier} is not supported, "
            f"supported tasks are {list(tasks.keys())}."
            )
    else:
        return tasks[identifier]


def load_tasks(
    tasks_to_run: List | None = None,
) -> Generator[Tuple[str, BenchmarkingTask]]:
    global tasks
    # Check whether all specified tasks exist or not.
    for t in tasks_to_run:
        if t not in tasks:
            raise ValueError(
                f"Task {t} is not supported, "
                f"supported tasks are {list(tasks.keys())}."
            )

    return [(task_name, get_task(task_name)) for task_name in tasks_to_run]
