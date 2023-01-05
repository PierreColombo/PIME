from pime.abstract_class.discrete_estimator import DiscreteEstimator
import torch
from torch import Tensor


class ABDivergence(DiscreteEstimator):
    """
    This is a class that implements the AB divergence between two discrete distributions.
    ABDivergences have been proposed in :cite:t:`cichocki2011generalized` and used to measure
    similarity between sentences among others (see :cite:t:`Colombo2022InfoLM`).

    
    :param name: Name of the divergence useful to save the results
    :type name: str
    :param alpha: Coefficient :math:`\\alpha` of the AB divergence
    :type alpha: float
    :param beta: Coefficient :math:`\\beta` of the AB divergence
    :type beta: float

    .. math::
        D_{\\alpha,\\beta}(P||Q) = \\frac{1}{\\alpha \\beta} \\left( \\sum_{i=1}^S \\frac{\\alpha}{\\alpha + \\beta} P_i^{\\alpha + \\beta} + \\sum_{i=1}^S \\frac{\\beta}{\\alpha + \\beta} Q_i^{\\alpha + \\beta} - \\sum_{i=1}^S P_i^{\\alpha} Q_i^{\\beta} \\right)
    """

    def __init__(self, name: str, alpha: float, beta: float):
        self.name = name
        self.alpha = alpha
        self.beta = beta
        assert self.alpha != 0
        assert self.beta != 0
        assert self.beta + self.alpha != 0

    def predict(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Predict divergence scores for the distributions.

        :param X: discrete input reference distribution over the discrete support
        :type X: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :param Y: discrete hypothesis reference distribution over  discrete support
        :type Y: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :return:  AB divergence between X and Y
        """

        first_term = torch.log(torch.sum(X ** (self.beta + self.alpha), dim=-1)) / (
                self.beta * (self.beta + self.alpha))
        second_term = torch.log(torch.sum(Y ** (self.beta + self.alpha), dim=-1)) / (
                self.alpha * (self.beta + self.alpha))
        third_term = torch.log(torch.sum((X ** (self.alpha)) * (Y ** (self.beta)), dim=-1)) / (self.beta * self.alpha)
        return first_term + second_term - third_term
