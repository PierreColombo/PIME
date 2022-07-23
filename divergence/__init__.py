import sys
sys.path.append('discrete/')
from divergence.ab_div import ABDivergence
from divergence.alpha_div import AlphaDivergence
from divergence.renyi_div import RenyiDivergence
from divergence.beta_div import BetaDivergence
from divergence.kl_div import KullbackLeiblerDivergence
from entropy import DiscreteEntropyEstimator
from divergence.lp import LP
from divergence.fisher_rao import FisherRao
from utils.symetrization import JeffreySymetrizationEstimator, JensenSymetrizationEstimator

DISCRETE_ESTIMATORS = {
    'ab_div': ABDivergence,
    'alpha_div': AlphaDivergence,
    'renyi_div': RenyiDivergence,
    'beta_div': BetaDivergence,
    'kl_div': KullbackLeiblerDivergence,
    'entropy': DiscreteEntropyEstimator,
    'lp': LP,
    'fisher_rao': FisherRao,
    'js_sym': JensenSymetrizationEstimator,
    'jf_sym': JeffreySymetrizationEstimator,

}
