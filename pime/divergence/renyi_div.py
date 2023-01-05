from pime.abstract_class.discrete_estimator import DiscreteEstimator
import torch
from torch import Tensor


class RenyiDivergence(DiscreteEstimator):
    """
    This is a class that implements the Rényi divergences between two discrete distributions.
    Rényi divergence was proposed by :cite:t:`renyi1961measures`.
    A thorough study was published by :cite:t:`VanErven2014Renyi`.
    It is used to measure similarity between sentences among others (see :cite:t:`Colombo2022InfoLM`).

    .. math::
        D_{\\alpha}(P||Q) = \\frac{1}{\\alpha - 1} \\log \\sum_{i=1}^S P_i^{\\alpha} Q_i^{1 - \\alpha}

    :param name: Name of the divergence useful to save the results
    :type name: str
    :param alpha: Coefficient :math:`\\alpha` of the Renyi Divergence
    :type alpha: float
    """

    def __init__(self, name: str, alpha: float):
        self.name = name
        self.alpha = alpha
        assert self.alpha != 1

    def predict(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Predict divergence scores for the distributions.

        :param X: discrete input reference distribution over the discrete support
        :type X: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :param Y: discrete hypothesis reference distribution over the discrete support
        :type Y: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :return:  Renyi divergence between X and Y
        """

        return torch.log(torch.sum(X ** self.alpha * Y ** (1 - self.alpha), dim=-1)) / (self.alpha - 1)
