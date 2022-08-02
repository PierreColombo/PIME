import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class TUBA(nn.Module):
    """
      This is a class that implements the estimator [15] to I(X,Y).
      :param x_dim: dimensions of samples from X
      :type x_dim:  int
      :param y_dim:dimensions of samples from Y
      :type y_dim: int
     :param hidden_size: the dimension of the hidden layer of the approximation network q(Y|X)
      :type hidden_size: int

      References
      ----------

      .. [15] Poole, B., Ozair, S., Van Den Oord, A., Alemi, A., and Tucker, G. On variational bounds of mutual
      information. In Chaudhuri, K. and Salakhutdinov, R. (eds.), Proceedings of the 36th International Conference on
      Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 5171–5180. PMLR, 09–15 Jun 2019.
    """

    def __init__(self, x_dim: int, y_dim: int, hidden_size: int):
        super(TUBA, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
        self.baseline = nn.Linear(y_dim, 1)

    def forward(self, x_samples: Tensor, y_samples: Tensor) -> Tensor:
        # shuffle and concatenate
        log_scores = self.baseline(y_samples)
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1))  # shape [sample_size, sample_size, 1]

        lower_bound = 1 + T0.mean() - log_scores.mean() - (
                (T1 - log_scores).logsumexp(dim=1) - np.log(sample_size)).exp().mean()
        return lower_bound

    def learning_loss(self, x_samples: Tensor, y_samples: Tensor) -> Tensor:
        return -self.forward(x_samples, y_samples)
