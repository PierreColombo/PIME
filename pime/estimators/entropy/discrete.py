import torch


class ShannonEntropy:

    def predict(self, X):

        return - (X * torch.log(X)).sum(-1)