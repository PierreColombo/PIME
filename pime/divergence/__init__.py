from pime.divergence.ab_div import ABDivergence
from pime.divergence.alpha_div import AlphaDivergence
from pime.divergence.beta_div import BetaDivergence
from pime.divergence.fisher_rao import FisherRao
from pime.divergence.kl_div import KullbackLeiblerDivergence
from pime.divergence.lp import LP
from pime.divergence.renyi_div import RenyiDivergence
from pime.entropy.discrete_entropy import DiscreteEntropyEstimator

DISCRETE_ESTIMATORS = {
    "alpha_div": AlphaDivergence,
    "beta_div": BetaDivergence,
    "ab_div": ABDivergence,
    "renyi_div": RenyiDivergence,
    "kl_div": KullbackLeiblerDivergence,
    "fisher_rao": FisherRao,
    "lp": LP,
    "entropy": DiscreteEntropyEstimator,
}


__all__ = [
    "ABDivergence",
    "AlphaDivergence",
    "BetaDivergence",
    "FisherRao",
    "KullbackLeiblerDivergence",
    "LP",
    "RenyiDivergence",
    "DISCRETE_ESTIMATORS",
]
