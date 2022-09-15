from ..abstract_class.mixed_estimator import MixedEstimator
from ..entropy.discret_entropy import DiscreteEntropyEstimator
from typing import Any
from copy import deepcopy
from torch import Tensor
import torch


class LinearEstimator(MixedEstimator):
    """
    This is a class that compute the Mutual information I(X;Y).
       In the special case where X and Y follows a gaussian multivariate distribution. This has been used in [19] to build fair classifiers and learn disentangle representations.

      :param name: name of the estimator
      :type x_dim:  str

    References
    ----------

      .. [20] Boudiaf, M., Ziko, I., Rony, J., Dolz, J., Piantanida, P., & Ben Ayed, I. (2020). Information maximization for few-shot learning. Advances in Neural Information Processing Systems, 33, 2445-2457.
    """

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
