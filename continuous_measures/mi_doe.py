import torch.nn as nn

from helper import PDF, ConditionalPDF


class MIDOE(nn.Module):
    def __init__(self, dim, hidden, pdf='gauss'):
        """
        paper from
        :param dim:
        :param hidden:
        :param pdf:
        """
        super(MIDOE, self).__init__()
        self.qY = PDF(dim, pdf)
        self.qY_X = ConditionalPDF(dim, hidden, pdf)

    def learning_loss(self, X, Y):
        hY = self.qY(Y)
        hY_X = self.qY_X(Y, X)

        loss = hY + hY_X
        return loss

    def forward(self, X, Y):
        hY = self.qY(Y)
        hY_X = self.qY_X(Y, X)

        return hY - hY_X
