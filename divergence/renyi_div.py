
from discrete_estimator import DiscreteEstimator
import torch

class RenyiDivergence(DiscreteEstimator):
    def __init__(self, name, alpha):
        self.name = name
        self.alpha = alpha
        assert self.alpha != 1

    def predict(self, X, Y):
        """
        :param ref_dist: discreate input reference distribution over the vocabulary
        :param hypo_dist: discreate hypothesis reference distribution over the vocabulary
        :return: fisher rao distance between the reference and hypothesis distribution
        """

        return torch.log(torch.sum(X ** self.alpha * Y ** (1 - self.alpha), dim=-1)) / (self.alpha - 1)

