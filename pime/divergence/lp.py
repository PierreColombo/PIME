import torch
from torch import Tensor

from pime.abstract_class.discrete_estimator import DiscreteEstimator


class LP(DiscreteEstimator):
    """
    This is a class that implements the :math:`\\ell_p`-norm between two discrete distributions.
    It has been used to measure similarity between sentences among others (see :cite:t:`Colombo2022InfoLM`).

    .. math::
        D_{p}(P||Q) = \\left( \\sum_{i=1}^S |P_i - Q_i|^p \\right)^{\\frac{1}{p}}

    :param name: Name of the KL divergence useful to save the results
    :type name: str
    :param power: Power of the norm
    :type power: float

    """

    def __init__(self, name, power):
        self.name = name
        self.power = power

    def predict(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Predict divergence scores for the distributions.

        :param X: discrete input reference distribution over the discrete support
        :type X: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :param Y: discrete hypothesis reference distribution over the discrete support
        :type Y: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :return:  :math:`\\ell_p`-norm between X and Y
        """
        return torch.norm(X - Y, p=self.power, dim=-1)
