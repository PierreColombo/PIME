from pime.abstract_class.discrete_estimator import DiscreteEstimator
import torch
from torch import Tensor


class KullbackLeiblerDivergence(DiscreteEstimator):
    """
    This is a class that implements the KL divergences between two discrete distributions.
    KLDivergence has been proposed in :cite:t:`shannon2001mathematical` and is used to measure similarity
    between sentences among others (see :cite:t:`Colombo2022InfoLM`).

    .. math::
        D_{KL}(P||Q) = \\sum_{i=1}^S P_i \\log \\frac{P_i}{Q_i}

    :param name: Name of the KL divergence useful to save the results
    :type name: str

    """

    def __init__(self, name: str):
        self.name = name

    def predict(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Predict divergence scores for the distributions.

        :param X: discrete input reference distribution over the discrete support
        :type X: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :param Y: discrete hypothesis reference distribution over the discrete support
        :type Y: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :return:  KL divergence between X and Y
        """
        return torch.sum(X * torch.log(X / Y), dim=1)
