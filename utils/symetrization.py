from divergence.discrete_estimator import DiscreteEstimator


class JeffreySymetrizationEstimator(DiscreteEstimator):
    def __init__(self, name, discret_estimator, **kwargs):
        self.name = name
        self.discret_estimator = discret_estimator(name, **kwargs)

    def predict(self, X, Y=None):
        """
        :param X: discrete input reference distribution over the vocabulary
        :param Y: discrete hypothesis reference distribution over the vocabulary
        :param alpha: alpha parameter of the divergence
        :return: alpha divergence between the reference and hypothesis distribution
        """
        return (self.discret_estimator.predict(X, Y) + self.discret_estimator.predict(Y, X)) / 2


class JensenSymetrizationEstimator(DiscreteEstimator):
    def __init__(self, name, discret_estimator, **kwargs):
        self.name = name
        self.discret_estimator = discret_estimator(name, **kwargs)

    def predict(self, X, Y=None):
        """
        :param X: discrete input reference distribution over the vocabulary
        :param Y: discrete hypothesis reference distribution over the vocabulary
        :param alpha: alpha parameter of the divergence
        :return: alpha divergence between the reference and hypothesis distribution
        """
        return (self.discret_estimator.predict(Y, (X + Y) / 2) + self.discret_estimator.predict(X, (X + Y) / 2)) / 2
