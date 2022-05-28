import torch
import math
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_divide(numerator, denominator):
    """
    :param numerator: quotient numerator
    :param denominator: quotient denominator
    :return: safe divide of numerator/denominator
    """
    return numerator / (denominator + 1e-30)


def nan_to_num(tensor):
    """
    :param tensor: input tensor
    :return: tensor without nan
    """
    tensor[tensor != tensor] = 0
    return tensor

def control_weights(model):
    def init_weights(m):
        if hasattr(m, 'weight') and hasattr(m.weight, 'uniform_') and True:
            torch.nn.init.uniform_(m.weight, a=-0.01, b=0.01)

    model.apply(init_weights)
