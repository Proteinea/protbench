import torch
import numpy as np
import random


def create_run_name(**kwargs):
    output = ""
    for k, v in kwargs.items():
        output += f"{k}={v}-"
    return output[:-1]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)