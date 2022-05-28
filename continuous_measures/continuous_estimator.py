from abc import ABC, abstractmethod


class ContinuousEstimator(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def fit(self, X, Y):
        pass

    @abstractmethod
    def predict(self, X, Y):
        pass
