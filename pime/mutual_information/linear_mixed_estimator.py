from ..abstract_class.mixed_estimator import MixedEstimator
from ..entropy.discrete_entropy import DiscreteEntropyEstimator
from typing import Any
from copy import deepcopy
from torch import Tensor
import torch


class LinearEstimator(MixedEstimator):
    """
    This is a class that compute the Mutual information I(X;Y).
    TODO: Provide explanation for what it does.

    :param classifier: Any classifier that implements the methods `fit` and `predict_proba`
    :type classifier: Classifier
    :param entropy_estimator: Estimator of (discrete) entropy. Should implement the method 'predict_proba'
    :type entropy_estimator: DiscreteEntropyEstimator

    """

    def __init__(self, classifier: Any, entropy_estimator: DiscreteEntropyEstimator):
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
