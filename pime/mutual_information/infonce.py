import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class InfoNCE(nn.Module):
    """
    This is a class that implements the estimator [14] to I(X,Y).
      InfoNCE is linked to pretraining of Neural Networks see [22,23,24].


      :param x_dim: dimensions of samples from X
      :type x_dim:  int
      :param y_dim:dimensions of samples from Y
      :type y_dim: int
     :param hidden_size: the dimension of the hidden layer of the approximation network q(Y|X)
      :type hidden_size: int

    References
    ----------

      .. [14] Oord, A. V. D., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748
      .. [22] Colombo, P., Chapuis, E., Labeau, M., & Clavel, C. (2021, November). Code-switched inspired losses for spoken dialog representations. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (pp. 8320-8337).
      .. [23]  Chapuis, E., Colombo, P., Manica, M., Labeau, M., & Clavel, C. (2020). Hierarchical pre-training for sequence labelling in spoken dialog. Findings of EMNLP 2020.
      .. [24] Kong, L., d'Autume, C. D. M., Ling, W., Yu, L., Dai, Z., & Yogatama, D. (2019). A mutual information maximization perspective of language representation learning. ICLR 2020.
    """

    def __init__(self, x_dim: int, y_dim: int, hidden_size: int):
        super(InfoNCE, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1),
                                    nn.Softplus())

    def forward(self, x_samples: Tensor, y_samples: Tensor) -> Tensor:
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1))  # [sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim=1).mean() - np.log(sample_size))
        return lower_bound

    def learning_loss(self, x_samples: Tensor, y_samples: Tensor) -> Tensor:
        return -self.forward(x_samples, y_samples)
