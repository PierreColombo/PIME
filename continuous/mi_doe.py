import torch.nn as nn

from helper import PDF, compute_negative_ln_prob
import torch


class FF_DOE(nn.Module):

    def __init__(self, dim_input, dim_output, dropout_rate=0):
        super(FF_DOE, self).__init__()
        self.residual_connection = False
        self.num_layers = 1
        self.layer_norm = True
        self.activation = 'tanh'
        self.stack = nn.ModuleList()
        for l in range(self.num_layers):
            layer = []

            if self.layer_norm:
                layer.append(nn.LayerNorm(dim_input))

            layer.append(nn.Linear(dim_input, dim_output))
            layer.append({'tanh': nn.Tanh(), 'relu': nn.ReLU()}[self.activation])
            layer.append(nn.Dropout(dropout_rate))

            self.stack.append(nn.Sequential(*layer))

        self.out = nn.Linear(dim_output, dim_output)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return self.out(x)


class ConditionalPDF(nn.Module):

    def __init__(self, x_dim, y_dim, pdf):
        super(ConditionalPDF, self).__init__()
        assert pdf in {'gauss', 'logistic'}
        self.dim = y_dim
        self.pdf = pdf
        self.X2Y = FF_DOE(x_dim, y_dim, 0.2)

    def forward(self, Y, X):
        mu, ln_var = torch.split(self.X2Y(X), self.dim, dim=1)
        cross_entropy = compute_negative_ln_prob(Y, mu, ln_var, self.pdf)
        return cross_entropy


class MIDOE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size, pdf='gauss'):
        """
        paper from
        :param dim:
        :param hidden:
        :param pdf:
        """
        super(MIDOE, self).__init__()
        self.qY = PDF(y_dim, pdf)
        self.qY_X = ConditionalPDF(x_dim, y_dim, pdf)

    def learning_loss(self, X, Y):
        hY = self.qY(Y)
        hY_X = self.qY_X(Y, X)

        loss = hY + hY_X
        return loss

    def forward(self, X, Y):
        hY = self.qY(Y)
        hY_X = self.qY_X(Y, X)

        return hY - hY_X
