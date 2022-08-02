from discrete_estimator import DiscreteEstimator
from torch import Tensor


class JeffreySymetrizationEstimator(DiscreteEstimator):
    """
    This is a class that implements the Jeffrey Symetrization trick applied to a divergence between two discrete
     distributions. See [9] for a detailled survey it has been used to measure similarity between sentences
      among others (see [2]).

    :param name: Name of the divergence usefull to save the results
    :type name: str
    :param discret_estimator: Estimator to symetrize
    :type discret_estimator: DiscreteEstimator
    :param kwargs: divergence parameters
    :type kwargs: dict

    References
    ----------

    .. [9] Basseville M. Divergence measures for statistical data processing—An annotated bibliography.
    Signal Process. 2013;93:621–633. doi: 10.1016/j.sigpro.2012.09.003.
    .. [2] Colombo, P. J. A., Clavel, C., & Piantanida, P. (2022, June). Infolm: A new metric to evaluate
    summarization & data2text generation. In Proceedings of the AAAI Conference on Artificial Intelligence
    (Vol. 36, No. 10, pp. 10554-10562).
    """

    def __init__(self, name, discret_estimator, **kwargs):
        self.name = name
        self.discret_estimator = discret_estimator(name, **kwargs)

    def predict(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Predict divergence scores for the distributions.

        :param X: discreate input reference distribution over the discret support
        :type X: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :param Y: discreate hypothesis reference distribution over the discret support
        :type Y: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :return:  divergence between X and Y symetrized using the Jensen Trick
        """
        return (self.discret_estimator.predict(X, Y) + self.discret_estimator.predict(Y, X)) / 2


class JensenSymetrizationEstimator(DiscreteEstimator):
    """
    This is a class that implements the Jensen Shannon Symetrization trick applied to a divergence between two discrete
     distributions. See [9] for a detailled survey it has been used to measure similarity between sentences
      among others (see [2]).

    :param name: Name of the divergence usefull to save the results
    :type name: str
    :param discret_estimator: Estimator to symetrize
    :param kwargs: divergence parameters
    :type kwargs: dict

    References
    ----------

    .. [9] Basseville M. Divergence measures for statistical data processing—An annotated bibliography.
    Signal Process. 2013;93:621–633. doi: 10.1016/j.sigpro.2012.09.003.
    .. [2] Colombo, P. J. A., Clavel, C., & Piantanida, P. (2022, June). Infolm: A new metric to evaluate
    summarization & data2text generation. In Proceedings of the AAAI Conference on Artificial Intelligence
    (Vol. 36, No. 10, pp. 10554-10562).
    """

    def __init__(self, name: str, discret_estimator: DiscreteEstimator, **kwargs):
        self.name = name
        self.discret_estimator = discret_estimator(name, **kwargs)

    def predict(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Predict divergence scores for the distributions.

        :param X: discreate input reference distribution over the discret support
        :type X: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :param Y: discreate hypothesis reference distribution over the discret support
        :type Y: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :return:  divergence between X and Y symetrized using the Jensen Trick
        """
        return (self.discret_estimator.predict(Y, (X + Y) / 2) + self.discret_estimator.predict(X, (X + Y) / 2)) / 2
