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


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, torch.NumberType):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def compute_cov(X):
    """
    :param X: input tensor of size N * hidden size
    :return: vairance covariance matrix
    """
    mean = compute_mean(X)
    return torch.matmul((X - mean).transpose(1, 0), X - mean)


def compute_mean(X):
    """
    :param X: input tensor of size N * hidden size
    :return: multivariate mean
    """
    return torch.mean(X, dim=0)


def control_weights(model):
    def init_weights(m):
        if hasattr(m, 'weight') and hasattr(m.weight, 'uniform_') and True:
            torch.nn.init.uniform_(m.weight, a=-0.01, b=0.01)

    model.apply(init_weights)
