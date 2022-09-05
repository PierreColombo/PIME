import torch.nn as nn
import torch
from utils import *
from mi_doe import MIDOE, ConditionalPDF
from entropy import KNIFE, ConditionalKNIFE

class MIKnife(nn.Module):
    """
      This is a class that implements the estimator [13] to I(X,Y).
      :param x_dim: dimensions of samples from X
      :type x_dim:  int
      :param y_dim:dimensions of samples from Y
      :type y_dim: int
     :param hidden_size: the dimension of the hidden layer of the approximation network q(Y|X)
      :type hidden_size: int

      References
      ----------

      .. [13] Cheng, P., Hao, W., Dai, S., Liu, J., Gan, Z., & Carin, L. (2020, November). Club: A contrastive
      log-ratio upper bound of mutual information. In International conference on machine learning (pp. 1779-1788). PMLR.
    """

    def __init__(self, x_dim: int, y_dim: int, hidden_size: int, pdf: str = 'gauss'):
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


class MIKernelEstimator(nn.Module):
    def __init__(self, device, number_of_samples, x_size, y_size,
                 # [K, d] to initialize the kernel :) so K is the number of points :)
                 average='fixed',  # un
                 cov_diagonal='var',  # diagonal of the covariance
                 cov_off_diagonal='var',  # var
                 use_joint=False):
        super(MIKernelEstimator, self).__init__()
        self.use_joint = use_joint
        self.count = 0
        self.count_learning = 0
        self.kernel_1 = KNIFE(device, number_of_samples, x_size)
        if self.use_joint:
            self.kernel_2 = KNIFE(device, number_of_samples, y_size)
            self.kernel_joint = KNIFE(device, number_of_samples, x_size + y_size)
        else:
            self.kernel_conditional = ConditionalKNIFE(device, number_of_samples, x_size, y_size)
        # self.kernel_joint.joint = True

    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        hz_1 = self.kernel_1(x_samples)

        # means = copy.deepcopy(self.kernel_joint.means.data.tolist())
        # self.kernel_joint.update_parameters(
        #    {1: self.kernel_1, 2: self.kernel_2})

        # assert means != copy.deepcopy(self.kernel_joint.means.data.tolist())
        if self.use_joint:
            hz_2 = self.kernel_2(y_samples)
            hz = self.kernel_joint(torch.cat([x_samples, y_samples], dim=1))
            self.count += 1
            return torch.abs(hz_1 + hz_2 - hz)  # abs to stay +
        else:
            hz_g1 = self.kernel_conditional(x_samples, y_samples)
            self.count += 1
            return torch.abs(hz_1 - hz_g1)

    def learning_loss(self, x_samples, y_samples):
        hz_1 = self.kernel_1(x_samples)
        if self.use_joint:
            hz_2 = self.kernel_2(y_samples)
            hz = self.kernel_joint(torch.cat([x_samples, y_samples], dim=1))
            self.count_learning += 1
            return hz_1 + hz_2 + hz
        else:
            hz_g1 = self.kernel_conditional(x_samples, y_samples)
            self.count_learning += 1
            return hz_1 + hz_g1
