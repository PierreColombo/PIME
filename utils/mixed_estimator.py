from abc import ABC, abstractmethod
from torch import Tensor


class MixedEstimator(ABC):
    """
    Abstract class to compute a similarity between discret and continuous distributions.

    :param name: Name of the divergence usefull to save the results
    :type name: str
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def predict(self, X: Tensor, Y: Tensor) -> Tensor:
        pass
