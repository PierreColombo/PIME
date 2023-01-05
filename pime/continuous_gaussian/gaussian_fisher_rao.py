import torch
from torch import Tensor

from pime.abstract_class.continuous_estimator import ContinuousEstimator

from ..misc.helpers import compute_cov, compute_mean


class FisherRao(ContinuousEstimator):
    """
    This is a class that compute the Fisher-Rao Distance
    in the special case where X and Y follow a multivariate Gaussian distribution.
    A geometrical interpretation of the Fisher-Rao distance if offered in :cite:t:`Costa2015Fisher`.
    This has been used in :cite:t:`Colombo2022Learning` to build fair classifiers and
    learn disentangled  representations.

    .. math ::

        m_1 = \\left( \\frac{\\left( \\mu_1 - \\mu_2 \\right)^2}{2} + \\left( \\sigma_{2} + \\sigma_{1} \\right)^2
        \\right)^{1/2}

        m_2 = \\left( \\frac{\\left( \\mu_1 - \\mu_2 \\right)^2}{2} + \\left( \\sigma_{2} - \\sigma_{1} \\right)^2
        \\right)^{1/2}

        D_{FR}(P||Q) = \\sqrt{2} \\left\\|\\left\\| \\log{\\left( \\frac{m_1 + m_2}{m_1 - m_2} \\right)}^2
        \\right\\|\\right\\|_2

    with :math:`\\sigma_{X}^{2} = \\text{diag}(\\Sigma_{X}) = \\text{diag}(\\text{Covariance}(X))`

    :param name: name of the estimator
    :type x_dim:  str

    """

    def __init__(self, name):
        self.name = name

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        :param X: Input distribution
        :type X: Tensor (B*hidden size)
        :param Y: Input distribution
        :type Y: Tensor (B*hidden size)
        :return:  Fisher Rao distance between the reference and hypothesis distribution under
                  the Multivariate Gaussian Hypothesis

        """
        # TODO : handle the case of 0

        self.ref_mean = compute_mean(X)
        self.ref_cov = compute_cov(X)
        self.hypo_mean = compute_mean(Y)
        self.hypo_cov = compute_cov(Y)

        first = (
            ((self.ref_mean - self.hypo_mean) ** 2) / 2
            + (torch.sqrt(torch.diag(self.hypo_cov)) + torch.sqrt(torch.diag(self.ref_cov))) ** 2
        ) ** (1 / 2)
        second = (
            ((self.ref_mean - self.hypo_mean) ** 2) / 2
            + (torch.sqrt(torch.diag(self.hypo_cov)) - torch.sqrt(torch.diag(self.ref_cov))) ** 2
        ) ** (1 / 2)
        rao = torch.sqrt(torch.sum((torch.log((first + second) / (first - second))) ** 2) * 2)
        return rao
