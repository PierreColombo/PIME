import torch
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../..')
sys.path.append('../../misc/')
sys.path.append('../examples/')
sys.path.append('../../examples/')
sys.path.append('../../mutual_information/')

from utils.utils import set_seed
from tqdm import tqdm
import time
import argparse
import logging
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker
from mutual_information import MI_CONTINUOUS_ESTIMATORS

logger = logging.getLogger(__name__)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


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
        y = torch.from_numpy(y).float().cuda()
    return x, y


def rho_to_mi(rho, dim):
    result = -dim / 2 * np.log(1 - rho ** 2)
    return result


def mi_to_rho(mi, dim):
    result = np.sqrt(1 - np.exp(-2 * mi / dim))
    return result


def main(cubic):
    args = get_args()
    logging.getLogger('matplotlib').setLevel(logging.WARN)
    logging.getLogger().addHandler(logging.StreamHandler())
    colors = sns.color_palette()

    args.cubic = cubic
    EMA_SPAN = 200
    settings = f"""
    Settings:
      CUDA:             {torch.cuda.is_available()!s}
      batchsize:        {args.batch_size!s}
      reps:             {args.reps!s}
      iterations:       {args.training_steps!s}
      kernel_samples:   {args.kernel_samples!s}
      d:                {args.dimension!s}
      """
    logger.info(settings)

    sample_dim = args.sample_dim
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    learning_rate = args.learning_rate
    training_steps = args.training_steps
    mi_list = [2 * i for i in range(args.max_mi)]
    model_list = list(MI_CONTINUOUS_ESTIMATORS.keys())  # TODO : Modify here
    total_steps = training_steps * len(mi_list)

    logger.info('Starting Training Models')
    for seed in range(args.reps):
        logger.info(f'Training with seed {seed} over !')
        set_seed(seed)

        #######################################
        ####### TRAINING THE ESTIMATORS #######
        #######################################

        mi_results = dict()
        for model_name in tqdm(model_list, 'Models'):
            model = MI_CONTINUOUS_ESTIMATORS[model_name](sample_dim, sample_dim, hidden_size).to(
                device)  # TODO : check that
            optimizer = torch.optim.Adam(model.parameters(), learning_rate)

            mi_est_values = []

            start_time = time.time()
            for mi_value in tqdm(mi_list, 'MI'):
                rho = mi_to_rho(mi_value, sample_dim)

                for _ in tqdm(range(training_steps), 'Training Loop', position=0, leave=True):
                    batch_x, batch_y = sample_correlated_gaussian(rho, dim=sample_dim, batch_size=batch_size,
                                                                  to_cuda=torch.cuda.is_available(), cubic=args.cubic)
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

                logger.info("finish training for %s with true MI value = %f" % (model.__class__.__name__, mi_value))
            end_time = time.time()
            time_cost = end_time - start_time
            logger.info("model %s average time cost is %f s" % (model_name, time_cost / total_steps))
            mi_results[model_name] = mi_est_values

        ######################################
        ####### DYNAMIC ESTIMATE PLOTS #######
        ######################################
        logger.info('Plotting DYNAMIC ESTIMATE PLOTS')
        ncols = len(model_list)
        nrows = 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.4 * nrows))
        axs = np.ravel(axs)

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

            ax = plt.gca()
            ax.xaxis.set_major_formatter(ticker.EngFormatter())
            plt.xticks(horizontalalignment="right")

        plt.gcf().tight_layout()
        plt.savefig(f'{args.saving_path}_mi_estimation.pdf', bbox_inches=None)

        logger.info('Plotting MSE, VAR and BIAS PLOTS')

        #######################################
        ####### MSE, VAR and BIAS PLOTS #######
        #######################################

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

        plt.style.use('default')

        colors = list(plt.rcParams['axes.prop_cycle'])
        col_idx = [2, 4, 5, 1, 3, 0, 6, 7]

        ncols = 1
        nrows = len(model_list)
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
            if i == 1:
                plt.yscale('log')
            if i == 2:
                plt.legend(loc='upper left', prop={'size': 12})
                plt.xlabel('MI Values', fontsize=15)

        plt.gcf().tight_layout()
        plt.savefig(f'{args.saving_path}_errors.pdf', bbox_inches='tight')

        logger.info('Everything is done!')


def get_args():
    parser = argparse.ArgumentParser(description='Train diff entropy')
    parser.add_argument('--saving_path', type=str, default='example', help='Repeat how ofter')
    parser.add_argument('--reps', type=int, default=1, help='Repeat how ofter')
    parser.add_argument('--max_mi', type=int, default=5, help='Maximum mutual information to estimate')
    parser.add_argument('--sample_dim', type=int, default=20, help='Dimension of the data')
    parser.add_argument('--hidden_size', type=int, default=15, help='Hidden size of the estimators')
    parser.add_argument('--batch_size', type=int, required=False, default=128,
                        help='Size of batches')
    parser.add_argument('--training_steps', type=int, required=False, default=5000,
                        help='Number of iterations per epoch')
    parser.add_argument('--kernel_samples', type=int, required=False, default=500,
                        help='Number samples used for kernel construction')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--cubic', action='store_true', default=False,
                        help='Cubed transform')
    parser.add_argument('--use_tanh', action='store_true', default=False,
                        help='Use Tanh for the variance')
    parser.add_argument('--dimension', type=int, required=False, default=5,
                        help='Data dimension')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    logging.getLogger('matplotlib').setLevel(logging.WARN)
    logging.getLogger().addHandler(logging.StreamHandler())
    main(False)
