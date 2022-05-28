from abstract_class import HybridEstimator
from discret_discret_similarity import DiscreteEntropyEstimator, AlphaEntropy


from typing import Any, Union
from copy import deepcopy
from torch import Tensor
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import math
import numpy as np


class LinearEstimator(HybridEstimator):

    def __init__(self, classifier: Any, entropy_estimator: DiscreteEntropyEstimator):
        """
        Parameters
        ----------
        classifier : Any
            Any classifier that implements the function fit and predict_proba
        entropy_estimator: DiscreteEntropyEstimator
            Estimator of (discrete) entropy. Should implement the method 'predict_proba'
        """
        assert hasattr(classifier, 'fit') and callable(getattr(classifier, 'fit'))
        assert hasattr(classifier, 'predict_proba') and callable(getattr(classifier, 'predict_proba'))
        self.initial_classifier = classifier
        self.classifier = deepcopy(classifier)
        self.entropy_estimator = entropy_estimator
        self.epsilon = 1e-9

    def predict(self, X: Tensor, Y: Tensor):

        self.classifier.fit(X.numpy(), Y.numpy())
        y_cond_probas = torch.from_numpy(self.classifier.predict_proba(X.numpy()))
        y_marg_probas = y_cond_probas.mean(0, keepdim=True)
        cond_ent = self.entropy_estimator.predict(y_cond_probas).mean(0)
        marg_ent = self.entropy_estimator.predict(y_marg_probas)
        return marg_ent - cond_ent


def unit_tests():

    n_dims = 20
    n_points = 1000
    n_classes = 10
    n_tries = 10
    classifier = LogisticRegression()
    entropy_estimator = AlphaEntropy(name='voldemort', alpha=2.0)

    for i in range(n_tries):

        # 1) General case : make sure that the mi value is in the good range
        mi_estimator = LinearEstimator(classifier, entropy_estimator)
        x = torch.randn(n_points, n_dims)
        y = torch.randint(n_classes, (n_points,))

        predicted_mi = mi_estimator.predict(x, y)

        assert predicted_mi >= 0 and predicted_mi <= math.log(n_classes)

        # 2) Make sure that the mi is 0 in a trivial case
        class TrivialClassifier:

            def __init__(self, num_classes: int):
                self.num_classes = num_classes

            def fit(self, *args, **kwargs):
                pass

            def predict_proba(self, X):
                return np.ones((X.shape[0], self.num_classes)) / self.num_classes

        mi_estimator = LinearEstimator(TrivialClassifier(n_classes), entropy_estimator)
        predicted_mi = mi_estimator.predict(x, y)
        assert math.isclose(predicted_mi.item(), 0., abs_tol=1e-10), predicted_mi.item()


if __name__ == '__main__':
    unit_tests()
