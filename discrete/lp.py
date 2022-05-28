from discrete_estimator import DiscreteEstimator
import torch



class LP(DiscreteEstimator):
    def __init__(self, name, power):
        self.name = name
        self.power = power

    def predict(self, X, Y):
        """
        :param ref_dist: discreate input reference distribution over the vocabulary
        :param hypo_dist: discreate hypothesis reference distribution over the vocabulary
        :return: l1 norm between the reference and hypothesis distribution
        """
        return torch.norm(X - Y, p=self.power, dim=-1)

