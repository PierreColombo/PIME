import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import utils


class L1OutUB(nn.Module):  # naive upper bound
    """
      This is a class that implements the estimator [14] to I(X,Y).
      :param x_dim: dimensions of samples from X
      :type x_dim:  int
      :param y_dim:dimensions of samples from Y
      :type y_dim: int
     :param hidden_size: the dimension of the hidden layer of the approximation network q(Y|X)
      :type hidden_size: int

      References
      ----------

      .. [14] Poole, B., Ozair, S., Van Den Oord, A., Alemi, A., and ucker, G. On variational bounds of mutual information.
    In ICML, 2019.
    """

    def __init__(self, x_dim: int, y_dim: int, hidden_size: int):
        super(L1OutUB, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples: Tensor, y_samples: Tensor) -> Tensor:
        batch_size = y_samples.shape[0]
        mu, logvar = self.get_mu_logvar(x_samples)

        positive = (- (mu - y_samples) ** 2 / 2. / logvar.exp() - logvar / 2.).sum(dim=-1)  # [nsample]

        mu_1 = mu.unsqueeze(1)  # [nsample,1,dim]
        logvar_1 = logvar.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)  # [1,nsample,dim]
        all_probs = (- (y_samples_1 - mu_1) ** 2 / 2. / logvar_1.exp() - logvar_1 / 2.).sum(
            dim=-1)  # [nsample, nsample]

        diag_mask = torch.ones([batch_size]).diag().unsqueeze(-1).to(self.device) * (-20.)
        negative = utils.log_sum_exp(all_probs + diag_mask, dim=0) - np.log(batch_size - 1.)  # [nsample]

        return (positive - negative).mean()

    def loglikeli(self, x_samples: Tensor, y_samples: Tensor) -> Tensor:
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples: Tensor, y_samples: Tensor) -> Tensor:
        return - self.loglikeli(x_samples, y_samples)
