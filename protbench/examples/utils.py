import random

import numpy as np
import torch


def create_run_name(**kwargs) -> str:
    output = ""
    for k, v in kwargs.items():
        output += f"{k}={v}-"
    return output[:-1]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
