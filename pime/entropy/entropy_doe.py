from ..misc.utils import compute_negative_ln_prob
import torch.nn as nn


class EntropyDoe(nn.Module):

    def __init__(self, zc_dim, pdf='gauss'):
        super(EntropyDoe, self).__init__()
        assert pdf in {'gauss', 'logistic'}
        self.dim = zc_dim
        self.pdf = pdf
        self.mu = nn.Embedding(1, self.dim)
        self.ln_var = nn.Embedding(1, self.dim)  # ln(s) in logistic

    def forward(self, Y):
        cross_entropy = compute_negative_ln_prob(Y, self.mu.weight,
                                                 self.ln_var.weight, self.pdf)
        return cross_entropy

    def logpdf(self, Y):
        return -compute_negative_ln_prob(Y, self.mu.weight,
                                         self.ln_var.weight, self.pdf, mean=False)
