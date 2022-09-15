import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


class KNIFE(nn.Module):
    """
    This is a class that implements the estimator [20] to H(X).

      :param x_dim: dimensions of samples from X
      :type x_dim:  int
      :param y_dim:dimensions of samples from Y
      :type y_dim: int
      :param hidden_size: the dimension of the hidden layer of the approximation network q(Y|X)
      :type hidden_size: int

    References
    ----------

    .. [20] Pichler, G., Colombo, P., Boudiaf, M., Koliander, G., & Piantanida, P. (2022). KNIFE: Kernelized-Neural Differential Entropy Estimation. ICML 2022.
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
            self.means = nn.Parameter(init_samples, requires_grad=True)  # [K, db]
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
