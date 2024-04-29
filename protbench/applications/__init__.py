from functools import partial
from typing import List
from typing import Optional

from peft import TaskType

from protbench.applications import models
from protbench.applications.benchmarking_task import BenchmarkingTask
from protbench.applications.deeploc import DeepLoc
from protbench.applications.fluorescence import Fluorescence
from protbench.applications.gb1_sampled import GB1Sampled
from protbench.applications.ppi import PPI
from protbench.applications.remote_homology import RemoteHomology
from protbench.applications.solubility import Solubility
from protbench.applications.ssp3 import SSP3
from protbench.applications.ssp8 import SSP8
from protbench.applications.thermostability import Thermostability


def get_tasks(tasks_to_run: Optional[List] = None, from_embeddings=False):
    tasks = {
        "ssp3_casp12": partial(
            SSP3,
            dataset="ssp3_casp12",
            from_embeddings=from_embeddings,
            task_type=TaskType.TOKEN_CLS,
        ),
        "ssp3_casp14": partial(
            SSP3,
            dataset="ssp3_casp14",
            from_embeddings=from_embeddings,
            task_type=TaskType.TOKEN_CLS,
        ),
        "ssp3_cb513": partial(
            SSP3,
            dataset="ssp3_cb513",
            from_embeddings=from_embeddings,
            task_type=TaskType.TOKEN_CLS,
        ),
        "ssp3_ts115": partial(
            SSP3,
            dataset="ssp3_ts115",
            from_embeddings=from_embeddings,
            task_type=TaskType.TOKEN_CLS,
        ),
        "ssp8_casp12": partial(
            SSP8,
            dataset="ssp8_casp12",
            from_embeddings=from_embeddings,
            task_type=TaskType.TOKEN_CLS,
        ),
        "ssp8_casp14": partial(
            SSP8,
            dataset="ssp8_casp14",
            from_embeddings=from_embeddings,
            task_type=TaskType.TOKEN_CLS,
        ),
        "ssp8_cb513": partial(
            SSP8,
            dataset="ssp8_cb513",
            from_embeddings=from_embeddings,
            task_type=TaskType.TOKEN_CLS,
        ),
        "ssp8_ts115": partial(
            SSP8,
            dataset="ssp8_ts115",
            from_embeddings=from_embeddings,
            task_type=TaskType.TOKEN_CLS,
        ),
        "deeploc": partial(
            DeepLoc,
            dataset="deeploc",
            from_embeddings=from_embeddings,
            task_type=TaskType.SEQ_CLS,
        ),
        "solubility": partial(
            Solubility,
            from_embeddings=from_embeddings,
            task_type=TaskType.SEQ_CLS,
        ),
        "remote_homology": partial(
            RemoteHomology,
            from_embeddings=from_embeddings,
            task_type=TaskType.SEQ_CLS,
        ),
        "fluorescence": partial(
            Fluorescence,
            from_embeddings=from_embeddings,
            task_type=TaskType.SEQ_CLS,
        ),
        "ppi": partial(
            PPI,
            from_embeddings=from_embeddings,
            task_type=TaskType.SEQ_CLS,
        ),
        "pli": partial(
            PLI,
            from_embeddings=from_embeddings,
            task_type=TaskType.SEQ_CLS,
        ),
        "gb1": partial(
            GB1Sampled,
            from_embeddings=from_embeddings,
            task_type=TaskType.SEQ_CLS,
        ),
        "thermostability": partial(
            Thermostability,
            from_embeddings=from_embeddings,
            task_type=TaskType.SEQ_CLS,
        ),
    }

    for task in tasks_to_run:
        if task not in tasks:
            raise ValueError(
                f"Task {task} is not supported, "
                f"supported tasks are {list(tasks.keys())}."
            )

    for task_name, task in tasks.items():
        if task_name not in tasks_to_run:
            continue
        yield task_name, task, task.keywords["task_type"]
