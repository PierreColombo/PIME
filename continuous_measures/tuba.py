import torch
import torch.nn as nn



class TUBA(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(TUBA, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
        self.baseline = nn.Linear(y_dim, 1)

    def forward(self, x_samples, y_samples):
        # shuffle and concatenate
        log_scores = self.baseline(y_samples)
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1))  # shape [sample_size, sample_size, 1]

        lower_bound = 1 + T0.mean() - log_scores.mean() - (
                (T1 - log_scores).logsumexp(dim=1) - np.log(sample_size)).exp().mean()
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)

