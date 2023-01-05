from torch import Tensor

from pime.abstract_class.discrete_estimator import DiscreteEstimator


class JeffreySymmetrizationEstimator(DiscreteEstimator):
    """
    This is a class that implements the Jeffrey Symmetrization trick applied to a divergence between
    two discrete distributions. See :cite:t:`basseville2013divergence` for a detailed survey it has been
    used to measure similarity between sentences among others (see :cite:t:`Colombo2022InfoLM`).

    :param name: Name of the divergence useful to save the results
    :type name: str
    :param discrete_estimator: Estimator to symmetrize
    :type discrete_estimator: DiscreteEstimator
    :param kwargs: divergence parameters
    :type kwargs: dict

    """

    def __init__(self, name, discrete_estimator, **kwargs):
        self.name = name
        self.discrete_estimator = discrete_estimator(name, **kwargs)

    def predict(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Predict divergence scores for the distributions.

        :param X: discrete input reference distribution over the discrete support
        :type X: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :param Y: discrete hypothesis reference distribution over the discrete support
        :type Y: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :return:  divergence between X and Y symmetrized using the Jensen Trick
        """
        return (self.discrete_estimator.predict(X, Y) + self.discrete_estimator.predict(Y, X)) / 2


class JensenSymmetrizationEstimator(DiscreteEstimator):
    """
    This is a class that implements the Jensen Shannon Symmetrization trick applied to a divergence
    between two discrete distributions. See :cite:t:`basseville2013divergence` for a detailed survey it has been
    used to measure similarity between sentences among others (see :cite:t:`Colombo2022InfoLM`).

    :param name: Name of the divergence useful to save the results
    :type name: str
    :param discrete_estimator: Estimator to symmetrize
    :param kwargs: divergence parameters
    :type kwargs: dict

    """

    def __init__(self, name: str, discrete_estimator: DiscreteEstimator, **kwargs):
        self.name = name
        self.discrete_estimator = discrete_estimator(name, **kwargs)

    def predict(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Predict divergence scores for the distributions.

        :param X: discrete input reference distribution over the discrete support
        :type X: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :param Y: discrete hypothesis reference distribution over the discrete support
        :type Y: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :return:  divergence between X and Y symmetrized using the Jensen Trick
        """
        return (self.discrete_estimator.predict(Y, (X + Y) / 2) + self.discrete_estimator.predict(X, (X + Y) / 2)) / 2
