from pimms.abstract_class.discrete_estimator import DiscreteEstimator
import torch
from torch import Tensor

class AlphaDivergence(DiscreteEstimator):
    """
    This is a class that implements the AB divergences between two discrete distributions.
      ABDivergences have been proposed in [3] and used to measure similarity between sentences among others (see [2]).

    :param name: Name of the divergence usefull to save the results
    :type name: str
    :param alpha: Coefficient $\alpha$ of the Alpha Divergence divergence
    :type alpha: float

    References
    ----------

    .. [3] Rényi, A. (1970). Probability Theory. North-Holland Publishing Company, Amsterdam.
    .. [2] Colombo, P. J. A., Clavel, C., & Piantanida, P. (2022, June). Infolm: A new metric to evaluate summarization & data2text generation. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 36, No. 10, pp. 10554-10562).
    """

    def __init__(self, name: str, alpha: float):
        self.name = name
        self.alpha = alpha
        assert alpha != 1 and alpha != 0

    def predict(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Predict divergence scores for the distributions.

        :param X: discreate input reference distribution over the discret support
        :type X: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :param Y: discreate hypothesis reference distribution over the discret support
        :type Y: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :return:  $\alpha$ divergence between X and Y
        """
        alpha = self.alpha

        return 1 / (alpha * (1 - alpha)) - torch.sum(X ** alpha * Y ** (1 - alpha), dim=-1) / (
                alpha * (1 - alpha))
