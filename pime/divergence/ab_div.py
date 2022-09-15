from pimms.abstract_class.discrete_estimator import DiscreteEstimator
import torch
from torch import Tensor


class ABDivergence(DiscreteEstimator):
    """
    This is a class that implements the AB divergences between two discrete distributions.
       ABDivergences have been proposed in [1] and used to measure similarity between sentences among others (see [2]).

    :param name: Name of the divergence usefull to save the results
    :type name: str
    :param alpha: Coefficient $\alpha$ of the AB divergence
    :type alpha: float
    :param beta: Coefficient $\beta$ of the AB divergence
    :type beta: float

    References
    ----------

    .. [1] Cichocki, A., Cruces, S., & Amari, S. I. (2011). Generalized alpha-beta divergences and their applicatio to robust nonnegative matrix factorization. Entropy, 13(1), 134-170.
    .. [2] Colombo, P. J. A., Clavel, C., & Piantanida, P. (2022, June). Infolm: A new metric to evaluate summarization & data2text generation. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 36, No. 10, pp. 10554-10562).
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

        :param X: discreate input reference distribution over the discret support
        :type X: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :param Y: discreate hypothesis reference distribution over  discret support
        :type Y: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :return:  ab divergence between X and Y
        """

        first_term = torch.log(torch.sum(X ** (self.beta + self.alpha), dim=-1)) / (
                self.beta * (self.beta + self.alpha))
        second_term = torch.log(torch.sum(Y ** (self.beta + self.alpha), dim=-1)) / (
                self.alpha * (self.beta + self.alpha))
        third_term = torch.log(torch.sum((X ** (self.alpha)) * (Y ** (self.beta)), dim=-1)) / (self.beta * self.alpha)
        return first_term + second_term - third_term


