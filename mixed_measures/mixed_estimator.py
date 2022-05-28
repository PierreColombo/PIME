from abc import ABC, abstractmethod


class MixedEstimator(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def predict(self, X, Y):
        pass
