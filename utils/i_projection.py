from divergence.discrete_estimator import DiscreteEstimator
import numpy as np

class IProjector(DiscreteEstimator):
    def __init__(self, name, discret_estimator, **kwargs):
        self.name = name
        self.discret_estimator = discret_estimator(name, **kwargs)

    def predict(self, X, Y):
        """
        :param X: discrete input reference distribution over the vocabulary
        :param Y: discrete hypothesis reference distribution over the vocabulary
        :param alpha: alpha parameter of the divergence
        :return: alpha divergence between the reference and hypothesis distribution
        """

        I_proj = np.zeros(X.shape)
        for i in range(len(X)):
            comp_Y = self.discret_estimator.predict(X[i], Y)
            I_proj[i] = comp_Y.min()
        return I_proj