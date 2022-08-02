from discrete_estimator import DiscreteEstimator
import torch
from torch import Tensor

class FisherRao(DiscreteEstimator):
    """
    This is a class that implements the AB divergences between two discrete distributions.
      ABDivergences have been proposed in [5] and used to measure similarity between sentences among others (see [2]).

    :param name: Name of the Fisher Roa distance usefull to save the results
    :type name: str

    References
    ----------

    .. [5] Picot, M., Messina, F., Boudiaf, M., Labeau, F., Ayed, I. B., & Piantanida, P. (2022). Adversarial
    Robustness via Fisher-Rao Regularization. IEEE Transactions on Pattern Analysis and Machine Intelligence.
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
        :return:  Fisher Rao distance between X and Y
        """
        fisher_rao = torch.clamp(
            torch.sum(torch.sqrt(X) * torch.sqrt(
                Y),
                      dim=-1), 0, 1)
        return 2 * torch.acos(fisher_rao)

