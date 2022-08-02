from discrete_estimator import DiscreteEstimator
import torch
from torch import Tensor
class BetaDivergence(DiscreteEstimator):
    """
    This is a class that implements the $\beta$-divergences between two discrete distributions.
      $\beta$-divergences have been proposed in [4] and used to measure similarity between sentences among others (see [2]).

    :param name: Name of the divergence usefull to save the results
    :type name: str
    :param beta: Coefficient $\beta$ of the Alpha Divergence divergence
    :type beta: float

    References
    ----------

    .. [4] A. Basu, I. R. Harris, N. L. Hjort, and M. C. Jones. Robust and efficient estimation by minimising a
    density power divergence. Biometrika, 85(3):549–559, Sep. 1998.
    .. [2] Colombo, P. J. A., Clavel, C., & Piantanida, P. (2022, June). Infolm: A new metric to evaluate
    summarization & data2text generation. In Proceedings of the AAAI Conference on Artificial Intelligence
    (Vol. 36, No. 10, pp. 10554-10562).
    """
    def __init__(self, name: str, beta:float):
        self.name = name
        self.beta = beta
        assert self.beta != -1
        assert self.beta != 0

    def predict(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Predict divergence scores for the distributions.

        :param X: discreate input reference distribution over the discret support
        :type X: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :param Y: discreate hypothesis reference distribution over the discret support
        :type Y: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :return:  $\beta$ divergence between X and Y
        """

        first_term = torch.log(torch.sum(X ** (self.beta + 1), dim=-1)) / (self.beta * (self.beta + 1))
        second_term = torch.log(torch.sum(Y ** (self.beta + 1), dim=-1)) / (self.beta + 1)
        third_term = torch.log(torch.sum(X * Y ** (self.beta), dim=-1)) / (self.beta)
        return first_term + second_term - third_term
