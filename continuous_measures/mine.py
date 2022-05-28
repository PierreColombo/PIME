
import torch
import torch.nn as nn



class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.T_func(torch.cat([x_samples, y_shuffle], dim=-1))

        lower_bound = T0.mean() - torch.log(T1.exp().mean())

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)

