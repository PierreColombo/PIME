import sys
sys.path.append('.')
from ab_div import ABDivergence
from alpha_div import AlphaDivergence
from renyi_div import RenyiDivergence
from beta_div import BetaDivergence
from kl_div import KullbackLeiblerDivergence
from entropy import DiscreteEntropyEstimator
from lp import LP
from fisher_rao import FisherRao
from symetrization import JeffreySymetrizationEstimator, JensenSymetrizationEstimator

DISCRETE_ESTIMATORS = {
    'ab_div': ABDivergence,
    'alpha_div': AlphaDivergence,
    'renyi_div': RenyiDivergence,
    'beta_div': BetaDivergence,
    'kl_div': KullbackLeiblerDivergence,
    'entropy': DiscreteEntropyEstimator,
    'lp_div': LP,
    'fisher_rao': FisherRao,
    'js_sym': JensenSymetrizationEstimator,
    'jf_sym': JeffreySymetrizationEstimator,

}
