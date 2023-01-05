import torch
from torch import Tensor

from pime.abstract_class.discrete_estimator import DiscreteEstimator


class BetaDivergence(DiscreteEstimator):
    """
    This is a class that implements the Beta divergences between two discrete distributions.
    :math:`\\beta`-divergences have been proposed in :cite:t:`basu1998robust` and used to measure similarity between
    sentences among others (see :cite:t:`Colombo2022InfoLM`).

    .. math::
        D_{\\beta}(P||Q) = \\sum_{i=1}^S \\frac{P_i^{\\beta + 1}}{\\beta(\\beta + 1)} +
        \\frac{Q_i^{\\beta + 1} }{\\beta + 1} - \\frac{P_i Q_i^{\\beta}}{\\beta}

    :param name: Name of the divergence useful to save the results
    :type name: str
    :param beta: Coefficient :math:`\\beta` of the Beta Divergence
    :type beta: float
    """

    def __init__(self, name: str, beta: float):
        self.name = name
        self.beta = beta
        assert self.beta != -1
        assert self.beta != 0

    def predict(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Predict divergence scores for the distributions.

        :param X: discrete input reference distribution over the discrete support
        :type X: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :param Y: discrete hypothesis reference distribution over the discrete support
        :type Y: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :return: beta divergence between X and Y
        """

        first_term = torch.sum(X ** (self.beta + 1), dim=-1) / (self.beta * (self.beta + 1))
        second_term = torch.sum(Y ** (self.beta + 1), dim=-1) / (self.beta + 1)
        third_term = torch.sum(X * Y ** (self.beta), dim=-1) / (self.beta)
        return first_term + second_term - third_term
