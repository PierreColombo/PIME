from discrete_estimator import DiscreteEstimator
import torch
from torch import Tensor


class KullbackLeiblerDivergence(DiscreteEstimator):
    """
    This is a class that implements the KL divergences between two discrete distributions.
      KLDivergence have been proposed in [5] and used to measure similarity between sentences among others (see [2]).

    :param name: Name of the KL divergence usefull to save the results
    :type name: str

    References
    ----------

    .. [6] Shannon, C. E. (1948). A mathematical theory of communication. The Bell system technical journal, 27(3),
    379-423.
    .. [2] Colombo, P. J. A., Clavel, C., & Piantanida, P. (2022, June). Infolm: A new metric to evaluate
    summarization & data2text generation. In Proceedings of the AAAI Conference on Artificial Intelligence
    (Vol. 36, No. 10, pp. 10554-10562).
    """

    def __init__(self, name: str):
        self.name = name

    def predict(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Predict divergence scores for the distributions.

        :param X: discreate input reference distribution over the discret support
        :type X: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :param Y: discreate hypothesis reference distribution over the discret support
        :type Y: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :return:  KL divergence between X and Y
        """
        return torch.sum(X * torch.log(X / Y), dim=1)
