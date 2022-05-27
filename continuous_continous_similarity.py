import numpy as np
import math

import torch
import torch.nn as nn
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import seaborn as sns
from tensorboardX import SummaryWriter
from continuous_estimator_helpers import *
from abstract_class import ContinuousEstimator
import torch.nn as nn
from utils import compute_mean, compute_cov
from geomloss import SamplesLoss
from continuous_self import *


################################################
####### Multivariate Gaussian Hypothesis #######
################################################

class MGHClosedJS(ContinuousEstimator):
    def __init__(self, name):
        self.name = name

    def predict(self, ref_dist, hypo_dist):
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
        d = self.var.size(1)
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

    def fit(self, ref_dist, hypo_dist):
        self.ref_mean = compute_mean(ref_dist)
        self.ref_cov = compute_cov(ref_dist)
        self.hypo_mean = compute_mean(hypo_dist)
        self.hypo_cov = compute_cov(hypo_dist)


class MGHClosedRAO(ContinuousEstimator):
    def __init__(self, name):
        self.name = name

    def predict(self, ref_dist, hypo_dist):
        """
        :param ref_dist: continuous input reference distribution
        :param hypo_dist: continuous hypothesis reference distribution
        :return:  Fisher Rao distance between the reference and hypothesis distribution under
        the Multivariate Gaussian Hypothesis
        """
        """
        https://www.sciencedirect.com/science/article/pii/S0166218X14004211
        """
        # TODO : handle the case of 0
        first = (((self.ref_mean - self.hypo_mean) ** 2) / 2 + (
                torch.sqrt(torch.diag(self.hypo_cov)) + torch.sqrt(torch.diag(self.ref_cov))) ** 2) ** (1 / 2)
        second = (((self.ref_mean - self.hypo_mean) ** 2) / 2 + (
                torch.sqrt(torch.diag(self.hypo_cov)) - torch.sqrt(torch.diag(self.ref_cov))) ** 2) ** (1 / 2)
        rao = torch.sqrt(torch.sum((torch.log((first + second) / (first - second))) ** 2) * 2)
        return rao

    def fit(self, ref_dist, hypo_dist):
        self.ref_mean = compute_mean(ref_dist)
        self.ref_cov = compute_cov(ref_dist)
        self.hypo_mean = compute_mean(hypo_dist)
        self.hypo_cov = compute_cov(hypo_dist)


class MGHClosedFRECHET(ContinuousEstimator):
    def __init__(self, name):
        self.name = name

    def predict(self, ref_dist, hypo_dist):
        """
        :param ref_dist: continuous input reference distribution
        :param hypo_dist: continuous hypothesis reference distribution
        :return:  Frechet distance between the reference and hypothesis distribution under
        the Multivariate Gaussian Hypothesis
        """
        var_0 = torch.diag(self.ref_cov)
        var_1 = torch.diag(self.hypo_cov)
        return torch.norm(self.ref_mean - self.hypo_mean, p=2) ** 2 + torch.sum(
            var_0 + var_1 - 2 * (var_0 * var_1) ** (1 / 2))

    def fit(self, ref_dist, hypo_dist):
        self.ref_mean = compute_mean(ref_dist)
        self.ref_cov = compute_cov(ref_dist)
        self.hypo_mean = compute_mean(hypo_dist)
        self.hypo_cov = compute_cov(hypo_dist)


#############################################
####### Mutual Information Estimators #######
#############################################


class DOE(nn.Module):

    def __init__(self, dim, hidden, pdf='gauss'):
        super(DOE, self).__init__()
        self.qY = PDF(dim, pdf)
        self.qY_X = ConditionalPDF(dim, hidden, pdf)

    def learning_loss(self, X, Y):
        hY = self.qY(Y)
        hY_X = self.qY_X(Y, X)

        loss = hY + hY_X
        return loss

    def forward(self, X, Y):
        hY = self.qY(Y)
        hY_X = self.qY_X(Y, X)

        return hY - hY_X


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim]
    '''

    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()

        positive = - (mu - y_samples) ** 2 / logvar.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (torch.abs(positive.sum(dim=-1) - negative.sum(dim=-1))).mean()
        return upper_bound / 2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


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


class NWJ(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(NWJ, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self, x_samples, y_samples):
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1)) - 1.  # shape [sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim=1) - np.log(sample_size)).exp().mean()
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


class InfoNCE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(InfoNCE, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1),
                                    nn.Softplus())

    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1))  # [sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim=1).mean() - np.log(sample_size))
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


class L1OutUB(nn.Module):  # naive upper bound
    def __init__(self, x_dim, y_dim, hidden_size):
        super(L1OutUB, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        batch_size = y_samples.shape[0]
        mu, logvar = self.get_mu_logvar(x_samples)

        positive = (- (mu - y_samples) ** 2 / 2. / logvar.exp() - logvar / 2.).sum(dim=-1)  # [nsample]

        mu_1 = mu.unsqueeze(1)  # [nsample,1,dim]
        logvar_1 = logvar.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)  # [1,nsample,dim]
        all_probs = (- (y_samples_1 - mu_1) ** 2 / 2. / logvar_1.exp() - logvar_1 / 2.).sum(
            dim=-1)  # [nsample, nsample]

        diag_mask = torch.ones([batch_size]).diag().unsqueeze(-1).to(self.device) * (-20.)
        negative = log_sum_exp(all_probs + diag_mask, dim=0) - np.log(batch_size - 1.)  # [nsample]

        return (positive - negative).mean()

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class VarUB(nn.Module):  # variational upper bound
    def __init__(self, x_dim, y_dim, hidden_size):
        super(VarUB, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):  # [nsample, 1]
        mu, logvar = self.get_mu_logvar(x_samples)
        return 1. / 2. * (mu ** 2 + logvar.exp() - 1. - logvar).mean()

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


# Estimating MI as a difference of Continous Entropy

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
        self.kernel_1 = MultiGaussKernelEE(device, number_of_samples, x_size)
        if self.use_joint:
            self.kernel_2 = MultiGaussKernelEE(device, number_of_samples, y_size)
            self.kernel_joint = MultiGaussKernelEE(device, number_of_samples, x_size + y_size)
        else:
            self.kernel_conditional = MultiGaussKernelCondEE(device, number_of_samples, x_size, y_size)
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


import torch.nn as nn
import torch
import numpy as np
from torch.distributions import MultivariateNormal


def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128, to_cuda=False, cubic=False):
    """Generate samples from a correlated Gaussian distribution."""
    mean = [0, 0]
    cov = [[1.0, rho], [rho, 1.0]]
    x, y = np.random.multivariate_normal(mean, cov, batch_size * dim).T

    x = x.reshape(-1, dim)
    y = y.reshape(-1, dim)

    if cubic:
        y = y ** 3

    if to_cuda:
        x = torch.from_numpy(x).float().cuda()
        # x = torch.cat([x, torch.randn_like(x).cuda() * 0.3], dim=-1)
        y = torch.from_numpy(y).float().cuda()
    return x, y


def rho_to_mi(rho, dim):
    result = -dim / 2 * np.log(1 - rho ** 2)
    return result


def mi_to_rho(mi, dim):
    result = np.sqrt(1 - np.exp(-2 * mi / dim))
    return result


import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(cubic=True):
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        set_seed(seed)
        lambda_ = 2

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        suffix = '9.07_{}_{}_{}'.format(cubic, lambda_, seed)

        sample_dim = 20
        batch_size = 128
        hidden_size = 15
        learning_rate = 0.001
        training_steps = 5000
        model_list = ["TUBA", "KNIFE"]  # CLUBSample

        mi_list = [2.0, 4.0, 6.0, 8.0, 10.0]  # , 12.0, 14.0, 16.0, 18.0, 20.0]

        total_steps = training_steps * len(mi_list)

        # train MI estimators with samples

        # train MI estimators with samples

        mi_results = dict()
        for model_name in tqdm(model_list, 'Models'):
            if model_name == 'Kernel_F':
                model = MIKernelEstimator(device, sample_dim // 2, sample_dim).to(device)
            elif model_name == 'KNIFE':
                model = MIKernelEstimator(device, batch_size // 6, sample_dim, sample_dim, use_joint=True).to(device)
            elif model_name == 'DOE':
                model = eval(model_name)(sample_dim, sample_dim).to(device)
            else:
                model = eval(model_name)(sample_dim, sample_dim, hidden_size).to(device)
            optimizer = torch.optim.Adam(model.parameters(), learning_rate)

            mi_est_values = []

            start_time = time.time()
            for mi_value in tqdm(mi_list, 'MI'):
                rho = mi_to_rho(mi_value, sample_dim)

                for step in tqdm(range(training_steps), 'Training Loop', position=0, leave=True):
                    batch_x, batch_y = sample_correlated_gaussian(rho, dim=sample_dim, batch_size=batch_size,
                                                                  to_cuda=torch.cuda.is_available(), cubic=cubic)
                    batch_x = torch.tensor(batch_x).float().to(device)
                    batch_y = torch.tensor(batch_y).float().to(device)
                    model.eval()
                    mi_est_values.append(model(batch_x, batch_y).item())

                    model.train()

                    model_loss = model.learning_loss(batch_x, batch_y)

                    optimizer.zero_grad()
                    model_loss.backward()
                    optimizer.step()

                    del batch_x, batch_y
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                print("finish training for %s with true MI value = %f" % (model.__class__.__name__, mi_value))
                # torch.save(model.state_dict(), "./model/%s_%d.pt" % (model.__class__.__name__, int(mi_value)))

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            end_time = time.time()
            time_cost = end_time - start_time
            print("model %s average time cost is %f s" % (model_name, time_cost / total_steps))
            mi_results[model_name] = mi_est_values

        import seaborn as sns
        import pandas as pd

        colors = sns.color_palette()

        EMA_SPAN = 200

        ncols = len(model_list)
        nrows = 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.4 * nrows))
        axs = np.ravel(axs)

        xaxis = np.array(list(range(total_steps)))
        yaxis_mi = np.repeat(mi_list, training_steps)

        for i, model_name in enumerate(model_list):
            plt.sca(axs[i])
            p1 = plt.plot(mi_results[model_name], alpha=0.4, color=colors[0])[0]  # color = 5 or 0
            plt.locator_params(axis='y', nbins=5)
            plt.locator_params(axis='x', nbins=4)
            mis_smooth = pd.Series(mi_results[model_name]).ewm(span=EMA_SPAN).mean()

            if i == 0:
                plt.plot(mis_smooth, c=p1.get_color(), label='$\\hat{I}$')
                plt.plot(yaxis_mi, color='k', label='True')
                plt.xlabel('Steps', fontsize=25)
                plt.ylabel('MI', fontsize=25)
                plt.legend(loc='upper left', prop={'size': 15})
            else:
                plt.plot(mis_smooth, c=p1.get_color())
                plt.yticks([])
                plt.plot(yaxis_mi, color='k')
                plt.xlabel('Steps', fontsize=25)

            plt.ylim(0, 15.5)
            plt.xlim(0, total_steps)
            plt.title(model_name, fontsize=35)
            import matplotlib.ticker as ticker

            ax = plt.gca()
            ax.xaxis.set_major_formatter(ticker.EngFormatter())
            plt.xticks(horizontalalignment="right")
            # plt.subplots_adjust( )

        plt.gcf().tight_layout()
        plt.savefig('mi_est_Gaussian_{}_copy.pdf'.format(suffix), bbox_inches=None)
        # plt.show()

        print('Second part')

        bias_dict = dict()
        var_dict = dict()
        mse_dict = dict()
        for i, model_name in tqdm(enumerate(model_list)):
            bias_list = []
            var_list = []
            mse_list = []
            for j in range(len(mi_list)):
                mi_est_values = mi_results[model_name][training_steps * (j + 1) - 500:training_steps * (j + 1)]
                est_mean = np.mean(mi_est_values)
                bias_list.append(np.abs(mi_list[j] - est_mean))
                var_list.append(np.var(mi_est_values))
                mse_list.append(bias_list[j] ** 2 + var_list[j])
            bias_dict[model_name] = bias_list
            var_dict[model_name] = var_list
            mse_dict[model_name] = mse_list

        # %%

        plt.style.use('default')  # ('seaborn-notebook')

        colors = list(plt.rcParams['axes.prop_cycle'])
        col_idx = [2, 4, 5, 1, 3, 0, 6, 7]

        ncols = 1
        nrows = 3
        fig, axs = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3. * nrows))
        axs = np.ravel(axs)

        for i, model_name in enumerate(model_list):
            plt.sca(axs[0])
            plt.plot(mi_list, bias_dict[model_name], label=model_name, marker='d', color=colors[col_idx[i]]["color"])

            plt.sca(axs[1])
            plt.plot(mi_list, var_dict[model_name], label=model_name, marker='d', color=colors[col_idx[i]]["color"])

            plt.sca(axs[2])
            plt.plot(mi_list, mse_dict[model_name], label=model_name, marker='d', color=colors[col_idx[i]]["color"])

        ylabels = ['Bias', 'Variance', 'MSE']
        for i in range(3):
            plt.sca(axs[i])
            plt.ylabel(ylabels[i], fontsize=15)

            if i == 0:
                if cubic:
                    plt.title('Cubic', fontsize=17)
                else:
                    plt.title('Gaussian', fontsize=17)
            if i == 1:
                plt.yscale('log')
            if i == 2:
                plt.legend(loc='upper left', prop={'size': 12})
                plt.xlabel('MI Values', fontsize=15)

        plt.gcf().tight_layout()
        plt.savefig('bias_variance_Gaussian_{}.pdf'.format(suffix), bbox_inches='tight')
        # plt.show()


if __name__ == '__main__':
    main()
