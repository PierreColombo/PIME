from pimms.abstract_class.continuous_estimator import ContinuousEstimator
from ..misc.helpers import compute_mean, compute_cov
import torch
from torch import Tensor


class JensenShannon(ContinuousEstimator):
    """
    This is a class that compute the JS divergence.
      In the special case where X and Y follows a gaussian multivariate distribution. This has been used in [19] to build fair classifiers and learn disentangle representations.

      :param name: name of the estimator
      :type x_dim:  str

    References
    ----------

      .. [19] Pierre Colombo, Guillaume Staerman, Nathan Noiry, and Pablo Piantanida. 2022. Learning Disentangled Textual Representations via Statistical Measures of Similarity. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2614–2630, Dublin, Ireland. Association for Computational Linguistics.
    """

    def __init__(self, name: str):
        self.name = name

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        :param X: Input distribution
        :type X: Tensor (B*hidden size)
        :param Y: Input distribution
        :type Y: Tensor (B*hidden size)
        :return:  Jensen-Shanon divergence between the reference and hypothesis distribution under
        the Multivariate Gaussian Hypothesis
        """
        """
        1/2[log|Σ2|/|Σ1|−𝑑+tr{Σ**0.5Σ1}+(𝜇2−𝜇1)𝑇Σ−12(𝜇2−𝜇1)]
        https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
        """
        self.ref_mean = compute_mean(X)
        self.ref_cov = compute_cov(X)
        self.hypo_mean = compute_mean(Y)
        self.hypo_cov = compute_cov(Y)

        d = self.ref_cov.size(1)
        var_0 = torch.diag(self.ref_cov)
        var_1 = torch.diag(self.hypo_cov)
        log_det_0_det_1 = (torch.sum(torch.log(var_0), dim=0) - torch.sum(torch.log(var_1), dim=0))
        log_det_1_det_0 = (torch.sum(torch.log(var_1), dim=0) - torch.sum(torch.log(var_0), dim=0))
        tr_0_1 = torch.sum(var_0 / var_1)
        tr_1_0 = torch.sum(var_1 / var_0)
        last_1 = torch.matmul((self.ref_mean - self.hypo_mean) * (var_1 ** (-1)), self.ref_mean - self.hypo_mean)
        last_0 = torch.matmul((self.ref_mean - self.hypo_mean) * (var_0 ** (-1)), self.ref_mean - self.hypo_mean)

        js = -2 * d + (log_det_0_det_1 + tr_1_0 + last_1 + log_det_1_det_0 + tr_0_1 + last_0)
        return js / 4
