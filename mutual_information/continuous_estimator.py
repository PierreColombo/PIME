from abc import ABC, abstractmethod


class ContinuousEstimator(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def forward(self, X, Y):
        pass
