from divergence.discrete_estimator import DiscreteEstimator
import torch
from torch import Tensor


class DiscreteEntropyEstimator(DiscreteEstimator):
    """
    This is a class that implements the Entropy of a discrete distribution. The entropy calculation is based on a
    divergence. Several cases are of interrest:
    1. if the divergence is the KL divergence and Y is uniform the Entropy is the Shannon entropy.
    2. if the divergence is the Renyi divergence and Y is uniform the Entropy is the Renyi Entropy.

    :param name: Name of the Entropy usefull to save the results
    :type name: str
    :param discret_estimator: divergence estimator used to compute the Entropy.
    :type discret_estimator: DiscreteEstimator

    """

    def __init__(self, name, discret_estimator, **kwargs):
        self.name = name
        self.discret_estimator = discret_estimator(name, **kwargs)

    def predict(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Predict divergence scores for the distributions.

        :param X: discreate input reference distribution over the vocabulary
        :type X: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :param Y: discreate hypothesis reference distribution over the vocabulary
        :type Y: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :return:  Entropy of X
        """
        if Y is not None:
            raise self.discret_estimator.predict(X, Y)
        else:
            batch_size = X.size(0)
            tensor_length = X.size(1)
            U = torch.tensor([1 / tensor_length] * tensor_length).unsqueeze(0).repeat(batch_size, 1).to(
                X.device)
            return self.discret_estimator.predict(X, U)
