import torch
import torch.nn as nn
import copy


class MultiGaussKernelEE(nn.Module):
    def __init__(self, device, number_of_samples, hidden_size,
                 # [K, d] to initialize the kernel :) so K is the number of points :)
                 average='weighted',  # un
                 cov_diagonal='var',  # diagonal of the covariance
                 cov_off_diagonal='var',  # var
                 ):

        self.K, self.d = number_of_samples, hidden_size
        super(MultiGaussKernelEE, self).__init__()
        self.device = device

        # base_samples.requires_grad = False
        # if kernel_positions in ('fixed', 'free'):
        #    self.mean = base_samples[None, :, :].to(self.args.device)
        # else:
        #    self.mean = base_samples[None, None, :, :].to(self.args.device)  # [1, 1, K, d]

        # self.K = K
        # self.d = d
        self.joint = False

        self.logC = torch.tensor([-self.d / 2 * np.log(2 * np.pi)]).to(
            self.device)

        self.means = nn.Parameter(torch.rand(number_of_samples, hidden_size), requires_grad=True).to(
            self.device)
        if cov_diagonal == 'const':
            diag = torch.ones((1, 1, self.d))
        elif cov_diagonal == 'var':
            diag = torch.ones((1, self.K, self.d))
        else:
            assert False, f'Invalid cov_diagonal: {cov_diagonal}'
        self.diag = nn.Parameter(diag.to(self.device))

        if cov_off_diagonal == 'var':
            tri = torch.zeros((1, self.K, self.d, self.d))
            self.tri = nn.Parameter(tri.to(self.device))
        elif cov_off_diagonal == 'zero':
            self.tri = None
        else:
            assert False, f'Invalid cov_off_diagonal: {cov_off_diagonal}'

        self.weigh = torch.ones((1, self.K), requires_grad=False).to(self.device)
        if average == 'weighted':
            self.weigh = nn.Parameter(self.weigh, requires_grad=True)
        else:
            assert average == 'fixed', f"Invalid average: {average}"

    def logpdf(self, x, y=None):
        assert len(x.shape) == 2 and x.shape[1] == self.d, 'x has to have shape [N, d]'
        x = x[:, None, :]
        w = torch.log_softmax(self.weigh, dim=1)
        y = x - self.means
        if self.tri is not None:
            y = y * self.diag + torch.squeeze(torch.matmul(torch.tril(self.tri, diagonal=-1), y[:, :, :, None]), 3)
        else:
            y = y * self.diag
        y = torch.sum(y ** 2, dim=2)

        y = -y / 2 + torch.sum(torch.log(torch.abs(self.diag)), dim=2) + w
        if self.joint:
            y = torch.log(torch.sum(torch.exp(y), dim=-1) / 2)
        else:
            y = torch.logsumexp(y, dim=-1)
        return self.logC + y

    def learning_loss(self, x_samples, y=None):
        return -self.forward(x_samples)

    def update_parameters(self, kernel_dict):
        tri = []
        means = []
        weigh = []
        diag = []
        for key, value in kernel_dict.items():  # detach and clone
            tri.append(copy.deepcopy(value.tri.detach().clone()))
            means.append(copy.deepcopy(value.means.detach().clone()))
            weigh.append(copy.deepcopy(value.weigh.detach().clone()))
            diag.append(copy.deepcopy(value.diag.detach().clone()))

        self.tri = nn.Parameter(torch.cat(tri, dim=1), requires_grad=True).to(self.device)
        self.means = nn.Parameter(torch.cat(means, dim=0), requires_grad=True).to(self.device)
        self.weigh = nn.Parameter(torch.cat(weigh, dim=-1), requires_grad=True).to(self.device)
        self.diag = nn.Parameter(torch.cat(diag, dim=1), requires_grad=True).to(self.device)

    def pdf(self, x):
        return torch.exp(self.logpdf(x))

    def forward(self, x, y=None):
        y = torch.abs(-self.logpdf(x))
        return torch.mean(y)


class MultiGaussKernelCondEE(nn.Module):

    def __init__(self, device,
                 number_of_samples,  # [K, d]
                 x_size, y_size,
                 layers=1,
                 ):
        super(MultiGaussKernelCondEE, self).__init__()
        self.K, self.d = number_of_samples, y_size
        self.device = device

        self.logC = torch.tensor([-self.d / 2 * np.log(2 * np.pi)]).to(self.device)

        # mean_weight = 10 * (2 * torch.eye(K) - torch.ones((K, K)))
        # mean_weight = _c(mean_weight[None, :, :, None])  # [1, K, K, 1]
        # self.mean_weight = nn.Parameter(mean_weight, requires_grad=True)

        self.std = FF(self.d, self.d * 2, self.K, layers)
        self.weight = FF(self.d, self.d * 2, self.K, layers)
        # self.mean_weight = FF(d, hidden, K**2, layers)
        self.mean_weight = FF(self.d, self.d * 2, self.K * x_size, layers)
        self.x_size = x_size

    def _get_mean(self, y):
        # mean_weight = self.mean_weight(y).reshape((-1, self.K, self.K, 1))  # [N, K, K, 1]
        # means = torch.sum(torch.softmax(mean_weight, dim=2) * self.base_X, dim=2)  #[1, K, d]
        means = self.mean_weight(y).reshape((-1, self.K, self.x_size))  # [N, K, d]
        return means

    def logpdf(self, x, y):  # H(X|Y)
        # for data in (x, y):
        # assert len(data.shape) == 2 and data.shape[1] == self.d, 'x has to have shape [N, d]'
        # assert x.shape == y.shape, "x and y need to have the same shape"

        x = x[:, None, :]  # [N, 1, d]

        w = torch.log_softmax(self.weight(y), dim=-1)  # [N, K]
        std = self.std(y).exp()  # [N, K]
        # std = self.std(y)  # [N, K]
        mu = self._get_mean(y)  # [1, K, d]

        y = x - mu  # [N, K, d]
        y = std ** 2 * torch.sum(y ** 2, dim=2)  # [N, K]

        y = -y / 2 + self.d * torch.log(torch.abs(std)) + w
        y = torch.logsumexp(y, dim=-1)
        return self.logC + y

    def pdf(self, x, y):
        return torch.exp(self.logpdf(x, y))

    def forward(self, x, y):
        z = -self.logpdf(x, y)
        return torch.mean(z)
