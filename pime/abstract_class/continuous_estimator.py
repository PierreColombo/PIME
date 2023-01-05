from abc import ABC, abstractmethod

from torch import Tensor


# TODO check that the documentation is fine
class ContinuousEstimator(ABC):
    """
    Interface for computing the similarity between two continuous distributions.

    :param name: Name of the divergence useful to save the results
    :type name: str
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        This method should take the two Tensors `X` and `Y` as input and produce as similarity
        score, which is returned.

        :param X: 1st input
        :type X: Tensor
        :param Y: 2nd input
        :param Y: Tensor
        :returns: similarity score
        :rtype: [1x1] Tensor
        """
        pass
