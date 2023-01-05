import torch
from torch import Tensor

from pime.abstract_class.discrete_estimator import DiscreteEstimator


class DiscreteEntropyEstimator(DiscreteEstimator):
    """
    This is a class that estimates the entropy of a discrete distribution based on a particular divergence
    estimate. The entropy calculation is based on a divergence. The divergence estimator can be passed as an argument.
    Two cases are of particular interest:

    * if the divergence is the KL divergence and Y is uniform, the divergence is the Shannon entropy.
    * if the divergence is the Rényi divergence and Y is uniform, the divergence is the Rényi Entropy.

    :param name: Name of the Entropy useful to save the results
    :type name: str
    :param discrete_estimator: divergence estimator used to compute the Entropy.
    :type discrete_estimator: DiscreteEstimator
    :param kwargs: Additional keyword arguments for `discrete_estimator`
    :type kwargs: dict
    """

    # TODO Given that this class takes X,Y, shouldn't it be a divergence estimator? But yet again,
    #      it essentially does nothing except for constructing a uniform distribution.

    def __init__(self, name, discrete_estimator, **kwargs):
        self.name = name
        self.discrete_estimator = discrete_estimator(name, **kwargs)

    def predict(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Predict divergence scores for the distributions.

        :param X: discrete input reference distribution over the vocabulary
        :type X: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :param Y: discrete hypothesis reference distribution over the vocabulary. [default: uniform distribution]
        :type Y: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :return: Entropy of X

        """
        # TODO: Behvior of this function is very strange. It raises a tensor. I don't think this
        #       is even allowed by python standards. Maybe we should remove the Y argument.
        if Y is not None:
            raise self.discrete_estimator.predict(X, Y)
        else:
            batch_size = X.size(0)
            tensor_length = X.size(1)
            U = torch.tensor([1 / tensor_length] * tensor_length).unsqueeze(0).repeat(batch_size, 1).to(X.device)
            return self.discrete_estimator.predict(X, U)
