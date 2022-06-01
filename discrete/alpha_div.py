from discrete_estimator import DiscreteEstimator
import torch


class AlphaDivergence(DiscreteEstimator):
    def __init__(self, name, alpha):
        self.name = name
        self.alpha = alpha
        assert alpha != 1 and alpha != 0

    def predict(self, X, Y):
        """
        :param X: discreate input reference distribution over the vocabulary
        :param Y: discreate hypothesis reference distribution over the vocabulary
        :param alpha: alpha parameter of the divergence
        :return: alpha divergence between the reference and hypothesis distribution
        """
        alpha = self.alpha

        return 1 / (alpha * (1 - alpha)) - torch.sum(X ** alpha * Y ** (1 - alpha), dim=-1) / (
                alpha * (1 - alpha))
