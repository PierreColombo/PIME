from discrete_estimator import DiscreteEstimator
import numpy as np
from torch import Tensor


class IProjector(DiscreteEstimator):
    """
       This is a class that implements the I-projection based on any kind of information measure.
       See [10] for a detailled survey it has been used to detect adversarial attacks among others (see [11]).

       :param name: Name of the divergence usefull to save the results
       :type name: str
       :param discret_estimator: Estimator to symetrize
       :param kwargs: divergence parameters
       :type kwargs: dict

       References
       ----------

       .. [10] Csiszár, I., & Matus, F. (2003). Information projections revisited. IEEE Transactions on Information
       Theory, 49(6), 1474-1490.
       .. [11] Picot, M, Piantanida, P. & Colombo, P (2022, July). An I-Projection-based Adversarial Attack Detector
       """

    def __init__(self, name: str, discret_estimator: DiscreteEstimator, **kwargs):
        self.name = name
        self.discret_estimator = discret_estimator(name, **kwargs)

    def predict(self, X: Tensor, S_Y: Tensor) -> Tensor:
        """
        Predict divergence scores for the distributions.

        :param X: discreate input reference distribution over the discret support
        :type X: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :param S_Y: set of discreate hypothesis reference distribution over the discret support
        :type S_Y: tensor of size (B*S) where B is the size of the batch and S the size of the support.
        :return:  I projection between X and set of reference S_Y
        """

        I_proj = np.zeros(X.shape)
        for i in range(len(X)):
            comp_Y = self.discret_estimator.predict(X[i], S_Y)
            I_proj[i] = comp_Y.min()
        return I_proj
