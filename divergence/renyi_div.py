from discrete_estimator import DiscreteEstimator
import torch
from torch import Tensor


class RenyiDivergence(DiscreteEstimator):
    """
    This is a class that implements the Renyi divergences between two discrete distributions.
      Renyi have been proposed in [8] andstudied in [7] and used to measure similarity between
       sentences among others (see [2]).

    :param name: Name of the divergence usefull to save the results
    :type name: str
    :param alpha: Coefficient $\alpha$ of the Renyi Divergence
    :type alpha: float

    References
    ----------

    .. [7] Van Erven, T., & Harremos, P. (2014). Rényi divergence and Kullback-Leibler divergence.
    IEEE Transactions on Information Theory, 60(7), 3797-3820.
    .. [8] A. Rényi, “On measures of entropy and information,” in Proc. 4th
Berkeley Symp. Math. Statist. and Probability, vol. 1. 1961, pp. 547–561.
    .. [2] Colombo, P. J. A., Clavel, C., & Piantanida, P. (2022, June). Infolm: A new metric to evaluate
    summarization & data2text generation. In Proceedings of the AAAI Conference on Artificial Intelligence
    (Vol. 36, No. 10, pp. 10554-10562).
    """

    def __init__(self, name: str, alpha: float):
        self.name = name
        self.alpha = alpha
        assert self.alpha != 1

    def predict(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Predict divergence scores for the distributions.

        :param X: discreate input reference distribution over the discret support
        :type X: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :param Y: discreate hypothesis reference distribution over the discret support
        :type Y: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :return:  Renyi divergence between X and Y
        """

        return torch.log(torch.sum(X ** self.alpha * Y ** (1 - self.alpha), dim=-1)) / (self.alpha - 1)
