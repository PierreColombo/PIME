import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class InfoNCE(nn.Module):
    """
    This is a class that implements the estimator for I(X,Y) from :cite:t:`nce`.

    InfoNCE is linked to pretraining of Neural Networks see :cite:t:`colombo2021code`,
    :cite:t:`chapuis2020hierarchical`, and :cite:t:`kong2020mutual`.


    :param x_dim: dimensions of samples from X
    :type x_dim:  int
    :param y_dim: dimensions of samples from Y
    :type y_dim: int
    :param hidden_size: the dimension of the hidden layer of the approximation network q(Y|X)
    :type hidden_size: int

    """

    def __init__(self, x_dim: int, y_dim: int, hidden_size: int):
        super(InfoNCE, self).__init__()
        self.F_func = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softplus(),
        )

    def forward(self, x_samples: Tensor, y_samples: Tensor) -> Tensor:
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1))  # [sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim=1).mean() - np.log(sample_size))
        return lower_bound

    def learning_loss(self, x_samples: Tensor, y_samples: Tensor) -> Tensor:
        return -self.forward(x_samples, y_samples)
