from .abstract_class import HybridEstimator
from typing import Any, Union
from copy import deepcopy
from torch import Tensor
import torch
from .discret_discret_similarity import DiscreteEstimator


class LinearEstimator(HybridEstimator):

    def __init__(self, classifier: Any, entropy_estimator: DiscreteEstimator):
        """
        Parameters
        ----------
        classifier : Any
            Any classifier that implements the function fit and predict_proba
        """
        assert hasattr(classifier, 'fit') and callable(getattr(classifier, 'fit'))
        assert hasattr(classifier, 'predict_proba') and callable(getattr(classifier, 'predict_proba'))
        self.initial_classifier = classifier
        self.classifier = deepcopy(classifier)
        self.entropy_estimator = entropy_estimator
        self.epsilon = 1e-9

    def reset(self):
        """
        Resets all attributes.
        """
        self.classifier = deepcopy(self.initial_classifier)

    def predict(self, X: Tensor, Y: Tensor):

        self.classifier.fit(X, Y)
        y_cond_probas = self.classifier.predict_proba(X)
        y_marg_probas = y_cond_probas.mean(0)
        cond_ent = self.entropy_estimator.predict(y_cond_probas).mean(0)
        marg_ent = self.entropy_estimator.predict(y_marg_probas)
        return marg_ent - cond_ent
