import torch.nn as nn
import torch
from torch import Tensor


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
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
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples: Tensor) -> tuple[Tensor, Tensor]:
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples: Tensor, y_samples: Tensor) -> Tensor:
        """

        Args:
            x_samples:
            y_samples:

        Returns:

        """
        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x_samples: Tensor, y_samples: Tensor):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples: Tensor, y_samples: Tensor):
        return - self.loglikeli(x_samples, y_samples)


class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
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
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples: Tensor)-> tuple[Tensor, Tensor]:
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples: Tensor, y_samples: Tensor) -> Tensor:
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def forward(self, x_samples: Tensor, y_samples: Tensor) -> Tensor:
        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()

        positive = - (mu - y_samples) ** 2 / logvar.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (torch.abs(positive.sum(dim=-1) - negative.sum(dim=-1))).mean()
        return upper_bound / 2.

    def learning_loss(self, x_samples: Tensor, y_samples: Tensor) -> Tensor:
        return - self.loglikeli(x_samples, y_samples)
