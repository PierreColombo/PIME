from pime.abstract_class.continuous_estimator import ContinuousEstimator
from ..misc.helpers import compute_mean, compute_cov
import torch
from torch import Tensor
class Frechet(ContinuousEstimator):
    """
    This is a class that compute the Frechet Distance.
    In the special case where X and Y follows a gaussian multivariate distribution.
    This has been used in :cite:t:`Colombo2022Learning` to build fair classifiers and learn disentangle representations.

      :param name: name of the estimator
      :type x_dim:  str

    """
    def __init__(self, name:str):
        self.name = name

    def forward(self, X:Tensor, Y:Tensor)->Tensor:
        """
        :param X: Input distribution
        :type X: Tensor (B*hidden size)
        :param Y: Input distribution
        :type Y: Tensor (B*hidden size)
        :return:  Frechet distance between the reference and hypothesis distribution under
                  the Multivariate Gaussian Hypothesis
        """

        self.ref_mean = compute_mean(X)
        self.ref_cov = compute_cov(X)
        self.hypo_mean = compute_mean(Y)
        self.hypo_cov = compute_cov(Y)

        var_0 = torch.diag(self.ref_cov)
        var_1 = torch.diag(self.hypo_cov)
        return torch.norm(self.ref_mean - self.hypo_mean, p=2) ** 2 + torch.sum(
            var_0 + var_1 - 2 * (var_0 * var_1) ** (1 / 2))
