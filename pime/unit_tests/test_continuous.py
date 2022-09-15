import unittest
from tqdm import tqdm
from mutual_information import MI_CONTINUOUS_ESTIMATORS
import torch


# 'gaussian_mi': MIGaussian,

class TestContinuousMethods(unittest.TestCase):

    def test_closed_forms(self):
        for measure_name in tqdm(['gaussian_frechet', 'gaussian_fisher_rao', 'gaussian_js'], 'Closed Form'):
            measure = MI_CONTINUOUS_ESTIMATORS[measure_name]
            batch_size = 100
            hidden_size = 50
            random_vectors = torch.rand(batch_size, hidden_size)
            closed_measure = measure('test_closed_form')
            distance = closed_measure.forward(random_vectors, random_vectors)
            assert distance == 0
