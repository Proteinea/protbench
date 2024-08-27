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
