import torch.nn as nn

from ..misc.utils import compute_negative_ln_prob


class EntropyDoe(nn.Module):
    """
    This class implements the entropy estimator of :cite:t:`mcallester2020formal`.

    :param zc_dim: ambient space dimension (:math:`d` in :cite:p:`mcallester2020formal`)
    :type zc_dim: int
    :param pdf: Kernel PDF; can be either "gauss" or "logistic"
    :type pdf: str
    """

    def __init__(self, zc_dim, pdf="gauss"):
        super(EntropyDoe, self).__init__()
        assert pdf in {"gauss", "logistic"}
        self.dim = zc_dim
        self.pdf = pdf
        self.mu = nn.Embedding(1, self.dim)
        self.ln_var = nn.Embedding(1, self.dim)  # ln(s) in logistic

    def forward(self, Y):
        cross_entropy = compute_negative_ln_prob(Y, self.mu.weight, self.ln_var.weight, self.pdf)
        return cross_entropy

    def logpdf(self, Y):
        return -compute_negative_ln_prob(Y, self.mu.weight, self.ln_var.weight, self.pdf, mean=False)
