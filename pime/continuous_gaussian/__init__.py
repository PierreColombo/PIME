from pime.continuous_gaussian.gaussian_fisher_rao import FisherRao
from pime.continuous_gaussian.gaussian_frechet import Frechet
from pime.continuous_gaussian.gaussian_js import JensenShannon

MI_CONTINUOUS_ESTIMATORS = {
    "gaussian_frechet": Frechet,
    "gaussian_fisher_rao": FisherRao,
    "gaussian_js": JensenShannon,
}

__all__ = ["JensenShannon", "Frechet", "FisherRao", "MI_CONTINUOUS_ESTIMATORS"]
