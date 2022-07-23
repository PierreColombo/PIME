from discrete_estimator import DiscreteEstimator
import torch

class BetaDivergence(DiscreteEstimator):
    def __init__(self, name, beta):
        self.name = name
        self.beta = beta
        assert self.beta != -1
        assert self.beta != 0

    def predict(self, X, Y):
        """
        :param ref_dist: discreate input reference distribution over the vocabulary
        :param hypo_dist: discreate hypothesis reference distribution over the vocabulary
        :return: beta divergence between the reference and hypothesis distribution
        """

        first_term = torch.log(torch.sum(X ** (self.beta + 1), dim=-1)) / (self.beta * (self.beta + 1))
        second_term = torch.log(torch.sum(Y ** (self.beta + 1), dim=-1)) / (self.beta + 1)
        third_term = torch.log(torch.sum(X * Y ** (self.beta), dim=-1)) / (self.beta)
        return first_term + second_term - third_term
