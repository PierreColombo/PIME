import unittest

from tqdm import tqdm

from discret_discret_similarity import *
from continuous_continous_similarity import *


class TestContinuousMethods(unittest.TestCase):

    def test_closed_forms(self):
        for measure in tqdm([MGHClosedJS, MGHClosedRAO, MGHClosedFRECHET], 'Closed Form'):
            batch_size = 100
            hidden_size = 50
            random_vectors = torch.rand(batch_size, hidden_size)
            closed_measure = measure('test_closed_form')
            closed_measure.fit(random_vectors, random_vectors)
            distance = closed_measure.predict(random_vectors, random_vectors)
            assert distance == 0
