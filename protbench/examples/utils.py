import random

import numpy as np
import torch
from omegaconf.listconfig import ListConfig


def create_run_name(**kwargs) -> str:
    output = ""
    for k, v in kwargs.items():
        if isinstance(v, (list, ListConfig)):
            v = "_".join(v)
        output += f"{k}_{v}-"
    return output[:-1]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def unpack_list_of_dicts(list_of_dicts):
    for model_family in list_of_dicts:
        for model_name, checkpoints in model_family.items():
            for checkpoint in checkpoints:
                yield model_name, checkpoint
