import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


class KNIFE(nn.Module):
    """
    This class implements the entropy estimator in :cite:t:`pichler2022differential`.

    :param zc_dim: ambient dimension (:math:`d` in :cite:p:`pichler2022differential`)
    :type zc_dim: int
    :param init_samples: optional samples for initializing the mean values :math:`b_m`
    :type init_samples: Tensor
    :param optimize_mu: if `False`, the `init_samples` are fixed and not optimized
    :type optimize_mu: bool
    :param batch_size: number of samples in `init_samples` (if given); only used if `optimize_mu` is `False`
    :type batch_size: int
    :param marg_modes: number of samples in `init_samples` (if given); only used if `optimize_mu` is `True`
    :type marg_modes: int
    :param use_tanh: if `True`, :math:`\\tanh()` is applied to the log-variance before exponentiation
    :type use_tanh: bool
    :param init_std: standard deviation for Gaussian initialization of parameters
    :type init_std: float
    :param cov_diagonal: if "var", the diagonal entries of the covariance matrices
                         :math:`A_1, A_2, \\dots, A_M` are considered training parameters;
                         otherwise they will not be trained.
    :type cov_diagonal: str
    :param cov_off_diagonal: if "var", the off-diagonal entries of the covariance matrices
                             :math:`A_1, A_2, \\dots, A_M` are considered training parameters;
                             otherwise they will not be trained.
    :type cov_off_diagonal: str
    :param average: if "var", the weights :math:`u_1, u_2, \\dots, u_M` are considered training parameters;
                    otherwise they will not be trained.
    :type average: str

    """

    def __init__(self, zc_dim: int, init_samples=None, optimize_mu: bool = True,
                 batch_size: int = 1, marg_modes: int = 128, use_tanh: bool = True, init_std: float = 0.001,
                 cov_diagonal: str = 'var', cov_off_diagonal: str = 'var', average: str = 'var'):
        self.optimize_mu = optimize_mu
        self.K = marg_modes if optimize_mu else batch_size
        self.d = zc_dim
        self.use_tanh = use_tanh
        self.init_std = init_std
        super(KNIFE, self).__init__()

        self.logC = torch.tensor([-self.d / 2 * np.log(2 * np.pi)])

        if init_samples is None:
            init_samples = self.init_std * torch.randn(self.K, self.d)
        # self.means = nn.Parameter(torch.rand(self.K, self.d), requires_grad=True)
        if self.optimize_mu:
            self.means = nn.Parameter(init_samples, requires_grad=True)  # [K, d]
        else:
            self.means = nn.Parameter(init_samples, requires_grad=False)

        if cov_diagonal == 'var':
            diag = self.init_std * torch.randn((1, self.K, self.d))
        else:
            diag = self.init_std * torch.randn((1, 1, self.d))
        self.logvar = nn.Parameter(diag, requires_grad=True)

        if cov_off_diagonal == 'var':
            tri = self.init_std * torch.randn((1, self.K, self.d, self.d))
            tri = tri.to(init_samples.dtype)
            self.tri = nn.Parameter(tri, requires_grad=True)
        else:
            self.tri = None

        weigh = torch.ones((1, self.K))
        if average == 'var':
            self.weigh = nn.Parameter(weigh, requires_grad=True)
        else:
            self.weigh = nn.Parameter(weigh, requires_grad=False)

    def logpdf(self, X: Tensor) -> Tensor:
        assert len(X.shape) == 2 and X.shape[1] == self.d, 'x has to have shape [N, d]'
        X = X[:, None, :]
        w = torch.log_softmax(self.weigh, dim=1)
        y = X - self.means
        logvar = self.logvar
        if self.use_tanh:
            logvar = logvar.tanh()
        var = logvar.exp()
        y = y * var
        # print(f"Marg : {var.min()} | {var.max()} | {var.mean()}")
        if self.tri is not None:
            y = y + torch.squeeze(torch.matmul(torch.tril(self.tri, diagonal=-1), y[:, :, :, None]), 3)
        y = torch.sum(y ** 2, dim=2)

        y = -y / 2 + torch.sum(torch.log(torch.abs(var) + 1e-8), dim=-1) + w
        y = torch.logsumexp(y, dim=-1)
        return self.logC.to(y.device) + y

    def update_parameters(self, Z: Tensor):
        self.means = Z

    def forward(self, X: Tensor) -> Tensor:
        y = -self.logpdf(X)
        return torch.mean(y)
