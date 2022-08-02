from utils.continuous_estimator import ContinuousEstimator
from utils.helper import compute_cov


class MIGaussian(ContinuousEstimator):
    """
      This is a class that compute the Mutual information I(X;Y). In the special case where X and Y follows a gaussian
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
        raise NotImplementedError
        self.name = name

    def predict(self, ref_dist, hypo_dist):
        """
        :param ref_dist: continuous input reference distribution
        :param hypo_dist: continuous hypothesis reference distribution
        :return:  MI between the reference and hypothesis distribution under
        the Multivariate Gaussian Hypothesis
        https://stats.stackexchange.com/questions/438607/mutual-information-between-subsets-of-variables-in-the-multivariate-normal-distr
        """
        raise NotImplemented

    def fit(self, ref_dist, hypo_dist):
        self.ref_cov = compute_cov(ref_dist)
        self.hypo_cov = compute_cov(hypo_dist)
