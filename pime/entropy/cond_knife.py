import torch
from ..misc.helpers import FF
import torch.nn as nn
import numpy as np
from torch import Tensor


class ConditionalKNIFE(nn.Module):
    """
    This class implements the estimator for :math:`H(X|Y)` from :cite:t:`pichler2022differential`.

    :param device: PyTorch Device
    :type device: torch.device
    :param number_of_samples: Number of samples used for the kernel (:math:`N` in :cite:p:`pichler2022differential`)
    :type number_of_samples: int
    :param x_size: dimensions of samples from X
    :type x_size:  int
    :param y_size: dimensions of samples from Y
    :type y_size: int
    :param hidden_size: the dimension of the hidden layer of the approximation network q(Y|X)
    :type hidden_size: int

    """

    def __init__(self, device,
                 number_of_samples,  # [K, d]
                 x_size, y_size,
                 layers=1,
                 ):
        super(ConditionalKNIFE, self).__init__()
        self.K, self.d = number_of_samples, y_size
        self.device = device

        self.logC = torch.tensor([-self.d / 2 * np.log(2 * np.pi)]).to(self.device)

        # mean_weight = 10 * (2 * torch.eye(K) - torch.ones((K, K)))
        # mean_weight = _c(mean_weight[None, :, :, None])  # [1, K, K, 1]
        # self.mean_weight = nn.Parameter(mean_weight, requires_grad=True)

        self.std = FF(self.d, self.d * 2, self.K, layers)
        self.weight = FF(self.d, self.d * 2, self.K, layers)
        # self.mean_weight = FF(d, hidden, K**2, layers)
        self.mean_weight = FF(self.d, self.d * 2, self.K * x_size, layers)
        self.x_size = x_size

    def _get_mean(self, Y: Tensor) -> Tensor:
        # mean_weight = self.mean_weight(y).reshape((-1, self.K, self.K, 1))  # [N, K, K, 1]
        # means = torch.sum(torch.softmax(mean_weight, dim=2) * self.base_X, dim=2)  #[1, K, d]
        means = self.mean_weight(Y).reshape((-1, self.K, self.x_size))  # [N, K, d]
        return means

    def logpdf(self, X: Tensor, Y: Tensor) -> Tensor:  # H(X|Y)
        # for data in (x, y):
        # assert len(data.shape) == 2 and data.shape[1] == self.d, 'x has to have shape [N, d]'
        # assert x.shape == y.shape, "x and y need to have the same shape"

        X = X[:, None, :]  # [N, 1, d]

        w = torch.log_softmax(self.weight(Y), dim=-1)  # [N, K]
        std = self.std(Y).exp()  # [N, K]
        # std = self.std(y)  # [N, K]
        mu = self._get_mean(Y)  # [1, K, d]

        y = X - mu  # [N, K, d]
        y = std ** 2 * torch.sum(y ** 2, dim=2)  # [N, K]

        y = -y / 2 + self.d * torch.log(torch.abs(std)) + w
        y = torch.logsumexp(y, dim=-1)
        return self.logC + y

    def pdf(self, X: Tensor, Y: Tensor) -> Tensor:
        return torch.exp(self.logpdf(X, Y))

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        z = -self.logpdf(X, Y)
        return torch.mean(z)
