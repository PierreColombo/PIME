import torch.nn as nn
import torch
from pime.entropy.knife import KNIFE
from pime.entropy.cond_knife import ConditionalKNIFE


class MIKnife(nn.Module):
    """
    This is a class that implements the estimator to I(X;Y) using the Kernel Estimator introduce in :cite:t:`pichler2022differential`.
    Two modes  are possible:
    * Using two Kernels to compute I(X;Y) = H(X) - H(X|Y)
    * Using three Kernels to compute  I(X;Y) = - H(X,Y) + H(X) + H(Y)

    :param x_size: dimensions of samples from X
    :type x_size: int
    :param y_size: dimensions of samples from Y
    :type y_size: int
    :param number_of_samples: number of samples used for the kernel
    :type number_of_samples: int

    """

    def __init__(self, x_size: int, y_size, number_of_samples=128,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 batch_size=128,
                 # [K, d] to initialize the kernel :) so K is the number of points :)
                 average='fixed',  # un
                 cov_diagonal='var',  # diagonal of the covariance
                 cov_off_diagonal='var',  # var
                 use_joint=True):
        super(MIKnife, self).__init__()
        self.use_joint = use_joint
        self.count = 0
        self.count_learning = 0
        self.kernel_1 = KNIFE(marg_modes=number_of_samples, zc_dim=x_size, batch_size=batch_size)
        if self.use_joint:
            self.kernel_2 = KNIFE(marg_modes=number_of_samples, zc_dim=y_size, batch_size=batch_size)
            self.kernel_joint = KNIFE(marg_modes=number_of_samples, zc_dim=x_size + y_size,
                                      batch_size=batch_size)
        else:
            self.kernel_conditional = ConditionalKNIFE(device=device, number_of_samples=number_of_samples,
                                                       x_size=x_size, y_size=y_size)

    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        hz_1 = self.kernel_1(x_samples)
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
