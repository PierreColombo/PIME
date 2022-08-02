import torch
import torch.nn as nn
from torch import Tensor


class MINE(nn.Module):
    """
      This is a class that implements the estimator [13] to I(X,Y).
      :param x_dim: dimensions of samples from X
      :type x_dim:  int
      :param y_dim:dimensions of samples from Y
      :type y_dim: int
     :param hidden_size: the dimension of the hidden layer of the approximation network q(Y|X)
      :type hidden_size: int

      References
      ----------

      .. [13] Cheng, P., Hao, W., Dai, S., Liu, J., Gan, Z., & Carin, L. (2020, November). Club: A contrastive
      log-ratio upper bound of mutual information. In International conference on machine learning (pp. 1779-1788). PMLR.
    """

    def __init__(self, x_dim: int, y_dim: int, hidden_size: int):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self, x_samples: Tensor, y_samples: Tensor) -> Tensor:
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.T_func(torch.cat([x_samples, y_shuffle], dim=-1))

        lower_bound = T0.mean() - torch.log(T1.exp().mean())

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound

    def learning_loss(self, x_samples: Tensor, y_samples: Tensor) -> Tensor:
        return -self.forward(x_samples, y_samples)
