from utils.continuous_estimator import ContinuousEstimator
from utils.helper import compute_cov


class MIGaussian(ContinuousEstimator):
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
