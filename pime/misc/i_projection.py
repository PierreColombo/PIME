from pime.abstract_class.discrete_estimator import DiscreteEstimator
import numpy as np
from torch import Tensor


class IProjector(DiscreteEstimator):
    """
    This is a class that implements the I-projection based on any kind of information measure.
    See :cite:t:`Csiszar2003Information` for a detailed survey it has been used to detect adversarial attacks
    among others (see [11]).

    :param name: Name of the divergence useful to save the results
    :type name: str
    :param discrete_estimator: Estimator to symmetrize
    :param kwargs: divergence parameters
    :type kwargs: dict

    References
    ----------
       .. [11] Picot, M, Piantanida, P. & Colombo, P (2022, July). An I-Projection-based Adversarial Attack Detector

    """
    # TODO: Where was [11] this published?

    def __init__(self, name: str, discrete_estimator: DiscreteEstimator, **kwargs):
        self.name = name
        self.discrete_estimator = discrete_estimator(name, **kwargs)

    def predict(self, X: Tensor, S_Y: Tensor) -> Tensor:
        """
        Predict divergence scores for the distributions.

        :param X: discrete input reference distribution over the discrete support
        :type X: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :param S_Y: set of discrete hypothesis reference distribution over the discrete support
        :type S_Y: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :return:  I projection between X and set of reference S_Y
        """

        I_proj = np.zeros(X.shape)
        for i in range(len(X)):
            comp_Y = self.discrete_estimator.predict(X[i], S_Y)
            I_proj[i] = comp_Y.min()
        return I_proj
