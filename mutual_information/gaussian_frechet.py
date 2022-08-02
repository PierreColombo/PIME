from utils.continuous_estimator import ContinuousEstimator
from utils.helper import compute_mean, compute_cov
import torch

class Frechet(ContinuousEstimator):
    """
      This is a class that compute the Frechet Distance. In the special case where X and Y follows a gaussian
      multivariate distribution. This has been used in [19] to build fair classifiers and learn disentangle
      representations.
      :param name: name of the estimator
      :type x_dim:  str

      References
      ----------

      .. [19] Pierre Colombo, Guillaume Staerman, Nathan Noiry, and Pablo Piantanida. 2022. Learning Disentangled
      Textual Representations via Statistical Measures of Similarity. In Proceedings of the 60th Annual Meeting of
      the Association for Computational Linguistics (Volume 1: Long Papers), pages 2614–2630, Dublin, Ireland.
      Association for Computational Linguistics.
    """
    def __init__(self, name):
        self.name = name

    def forward(self, ref_dist, hypo_dist):
        """
        :param ref_dist: continuous input reference distribution
        :param hypo_dist: continuous hypothesis reference distribution
        :return:  Frechet distance between the reference and hypothesis distribution under
        the Multivariate Gaussian Hypothesis
        """

        self.ref_mean = compute_mean(ref_dist)
        self.ref_cov = compute_cov(ref_dist)
        self.hypo_mean = compute_mean(hypo_dist)
        self.hypo_cov = compute_cov(hypo_dist)

        var_0 = torch.diag(self.ref_cov)
        var_1 = torch.diag(self.hypo_cov)
        return torch.norm(self.ref_mean - self.hypo_mean, p=2) ** 2 + torch.sum(
            var_0 + var_1 - 2 * (var_0 * var_1) ** (1 / 2))
