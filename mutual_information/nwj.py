import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class NWJ(nn.Module):
    """
      This is a class that implements the estimator [16] to I(X,Y).
      :param x_dim: dimensions of samples from X
      :type x_dim:  int
      :param y_dim:dimensions of samples from Y
      :type y_dim: int
     :param hidden_size: the dimension of the hidden layer of the approximation network q(Y|X)
      :type hidden_size: int

      References
      ----------

      .. [16] Nguyen, X., Wainwright, M. J., and Jordan, M. I. Estimating divergence functionals and the
      likelihood ratio by convex risk minimization. IEEE Transactions on Information Theory, 2010.
    """

    def __init__(self, x_dim: int, y_dim: int, hidden_size: int):
        super(NWJ, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self, x_samples: Tensor, y_samples: Tensor) -> Tensor:
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1)) - 1.  # shape [sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim=1) - np.log(sample_size)).exp().mean()
        return lower_bound

    def learning_loss(self, x_samples: Tensor, y_samples: Tensor) -> Tensor:
        return -self.forward(x_samples, y_samples)
