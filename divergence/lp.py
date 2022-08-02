from discrete_estimator import DiscreteEstimator
import torch
from torch import Tensor


class LP(DiscreteEstimator):
    """
    This is a class that implements the LP norms between two discrete distributions.
      It has been used to measure similarity between sentences among others (see [2]).

    :param name: Name of the KL divergence usefull to save the results
    :type name: str
    :param power: Power of the norm
    :type power: float

    References
    ----------

    .. [2] Colombo, P. J. A., Clavel, C., & Piantanida, P. (2022, June). Infolm: A new metric to evaluate
    summarization & data2text generation. In Proceedings of the AAAI Conference on Artificial Intelligence
    (Vol. 36, No. 10, pp. 10554-10562).
    """

    def __init__(self, name, power):
        self.name = name
        self.power = power

    def predict(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Predict divergence scores for the distributions.

        :param X: discreate input reference distribution over the discret support
        :type X: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :param Y: discreate hypothesis reference distribution over the discret support
        :type Y: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :return:  LP norm between X and Y
        """
        return torch.norm(X - Y, p=self.power, dim=-1)
