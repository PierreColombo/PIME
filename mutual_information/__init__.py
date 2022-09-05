import sys
sys.path.append('continuous/')
from entropy import ConditionalKNIFE
from club import CLUB, CLUBSample
from gaussian_fisher_rao import FisherRao
from gaussian_frechet import Frechet
from gaussian_js import JensenShannon
from infonce import InfoNCE
from entropy import KNIFE
from l1out import L1OutUB
from mi_doe import MIDOE
from mi_knife import MIKnife
from mine import MINE
from nwj import NWJ
from tuba import TUBA
from varub import VarUB

CONTINUOUS_ESTIMATORS = {
    'club': CLUB,
    'club_sample': CLUBSample,
    'cond_knife': ConditionalKNIFE,
    'gaussian_frechet': Frechet,
    'gaussian_fisher_rao': FisherRao,
    'gaussian_js': JensenShannon,
    'infonce': InfoNCE,
    'knife': KNIFE,
    'l1out': L1OutUB,
    'mi_doe': MIDOE,
    'mi_knife': MIKnife,
    'mine': MINE,
    'nwj': NWJ,
    'tuba': TUBA,
    'varub': VarUB,

}
