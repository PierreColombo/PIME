import torch
from torch import Tensor

from pime.abstract_class.discrete_estimator import DiscreteEstimator


class FisherRao(DiscreteEstimator):
    """
    This is a class that implements the Fisher-Rao divergences between two discrete distributions.
    Fisher Rao for regularization has been proposed in :cite:t:`picot2022adversarial` and used to measure
    similarity between sentences among others (see :cite:t:`Colombo2022InfoLM`).

    .. math::
        D_{FR}(P||Q) = 2 \\arccos \\left( \\sum_{i=1}^S \\sqrt{P_i} \\sqrt{Q_i} \\right)

    :param name: Name of the Fisher-Rao distance useful to save the results
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
        :return:  Fisher-Rao distance between X and Y
        """
        fisher_rao = torch.clamp(torch.sum(torch.sqrt(X) * torch.sqrt(Y), dim=-1), 0, 1)
        return 2 * torch.acos(fisher_rao)
