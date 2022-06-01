from discrete_estimator import DiscreteEstimator
import torch

class KullbackLeiblerDivergence(DiscreteEstimator):
    def __init__(self, name):
        self.name = name

    def predict(self, X, Y):
        """
        :param ref_dist: discreate input reference distribution over the vocabulary
        :param hypo_dist: discreate hypothesis reference distribution over the vocabulary
        :return: fisher rao distance between the reference and hypothesis distribution
        """
        return torch.sum(X * log(X / Y), dim=1)


