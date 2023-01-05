import torch
from torch import Tensor

from pime.abstract_class.discrete_estimator import DiscreteEstimator


class AlphaDivergence(DiscreteEstimator):
    """
    This is a class that implements the Alpha divergences between two discrete distributions.
    Alpha divergences have been introduced in :cite:t:`Renyi2007Probability` and used to measure
    similarity between sentences among others (see :cite:t:`Colombo2022InfoLM`).

    :param name: Name of the divergence useful to save the results
    :type name: str
    :param alpha: Coefficient :math:`\\alpha` of the Alpha divergence
    :type alpha: float

    .. math::
        D_{\\alpha}(P||Q) = \\frac{1}{\\alpha (1-\\alpha)}  \\left( 1 - \\sum_{i=1}^S P_i^{\\alpha} Q_i^{1-\\alpha} \\right)


    """

    def __init__(self, name: str, alpha: float):
        self.name = name
        self.alpha = alpha
        assert alpha != 1 and alpha != 0

    def predict(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Predict divergence scores for the distributions.

        :param X: discrete input reference distribution over the discrete support
        :type X: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :param Y: discrete hypothesis reference distribution over the discrete support
        :type Y: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :return: alpha divergence between X and Y
        """
        alpha = self.alpha

        return 1 / (alpha * (1 - alpha)) - torch.sum(X**alpha * Y ** (1 - alpha), dim=-1) / (alpha * (1 - alpha))
