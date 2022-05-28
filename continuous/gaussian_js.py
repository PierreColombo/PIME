from continuous_estimator import ContinuousEstimator
from helper import compute_mean, compute_cov
import torch


class JensenShannon(ContinuousEstimator):
    def __init__(self, name):
        self.name = name

    def forward(self, ref_dist, hypo_dist):
        """
        :param ref_dist: continuous input reference distribution
        :param hypo_dist: continuous hypothesis reference distribution
        :return:  Jensen-Shanon divergence between the reference and hypothesis distribution under
        the Multivariate Gaussian Hypothesis
        """
        """
        1/2[log|Σ2|/|Σ1|−𝑑+tr{Σ**0.5Σ1}+(𝜇2−𝜇1)𝑇Σ−12(𝜇2−𝜇1)]
        https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
        """
        self.ref_mean = compute_mean(ref_dist)
        self.ref_cov = compute_cov(ref_dist)
        self.hypo_mean = compute_mean(hypo_dist)
        self.hypo_cov = compute_cov(hypo_dist)

        d = self.ref_cov.size(1)
        var_0 = torch.diag(self.ref_cov)
        var_1 = torch.diag(self.hypo_cov)
        log_det_0_det_1 = (torch.sum(torch.log(var_0), dim=0) - torch.sum(torch.log(var_1), dim=0))
        log_det_1_det_0 = (torch.sum(torch.log(var_1), dim=0) - torch.sum(torch.log(var_0), dim=0))
        tr_0_1 = torch.sum(var_0 / var_1)
        tr_1_0 = torch.sum(var_1 / var_0)
        last_1 = torch.matmul((self.ref_mean - self.hypo_mean) * (var_1 ** (-1)), self.ref_mean - self.hypo_mean)
        last_0 = torch.matmul((self.ref_mean - self.hypo_mean) * (var_0 ** (-1)), self.ref_mean - self.hypo_mean)

        js = -2 * d + (log_det_0_det_1 + tr_1_0 + last_1 + log_det_1_det_0 + tr_0_1 + last_0)
        return js / 4

