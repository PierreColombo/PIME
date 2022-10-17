from typing import Tuple

import torch.nn as nn
from torch import Tensor


class VarUB(nn.Module):
    """
    This is a class that implements the estimator for I(X,Y) in :cite:t:`alemi2017deep`.

    :param x_dim: dimensions of samples from X
    :type x_dim:  int
    :param y_dim: dimensions of samples from Y
    :type y_dim: int
    :param hidden_size: the dimension of the hidden layer of the approximation network q(Y|X)
    :type hidden_size: int

    """

    # TODO Where is this mentioned in alemi2017deep ? I don't find this estimator.

    def __init__(self, x_dim: int, y_dim: int, hidden_size: int):
        super(VarUB, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples: Tensor) -> Tuple[Tensor, Tensor]:
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples: Tensor, y_samples: Tensor) -> Tensor:
        mu, logvar = self.get_mu_logvar(x_samples)
        return 1. / 2. * (mu ** 2 + logvar.exp() - 1. - logvar).mean()

    def loglikeli(self, x_samples: Tensor, y_samples: Tensor) -> Tensor:
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples: Tensor, y_samples: Tensor) -> Tensor:
        return - self.loglikeli(x_samples, y_samples)
