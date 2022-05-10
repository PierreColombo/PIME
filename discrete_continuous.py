from .abstract_class import HybridEstimator
from typing import Any, Union
from copy import deepcopy
from torch import Tensor
import torch


class LinearEstimator(HybridEstimator):

    def __init__(self, classifier: Any):
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
        self.epsilon = 1e-9

    def reset(self):
        """
        Resets all attributes.
        """
        self.classifier = deepcopy(self.initial_classifier)

    def predict(self, X: Tensor, Y: Tensor):

        self.classifier.fit(X, Y)
        y_probas = self.classifier.predict_proba(X)
        cond_ent = - (y_probas * torch.log(y_probas + self.epsilon)).sum(-1)
        marg_prob = y_probas.mean(0)
        marg_ent = - (marg_prob * torch.log(marg_prob + self.epsilon)).sum()
        return marg_ent - cond_ent
