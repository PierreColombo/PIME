import unittest
from discret_discret_similarity import *


class TestDiscreteMethods(unittest.TestCase):

    def test_alpha(self):
        for alpha in [-2, -1.01, -0.99, -0.01, 0.01, 0.99, 1.01, 2]:
            batch_size = 10
            tensor_length = 4
            uniform_tensor = torch.tensor([1 / tensor_length for _ in range(tensor_length)])
            batched_uniform_tensor = uniform_tensor.unsqueeze(0).repeat(batch_size, 1)
            random_vector = torch.nn.Softmax(dim=-1)(torch.rand(batch_size, tensor_length))
            alpha_div = AlphaDivergence('test_alpha_div', alpha)
            alpha_entropy = AlphaEntropy('test_alpha_div', alpha)
            0 = alpha_div.fit(random_vector,random_vector)
            0 = alpha_div.fit(random_vector,batched_uniform_tensor)
            0 = alpha_entropy.fit(random_vector)
            0 = alpha_entropy.fit(batched_uniform_tensor)
        self.assertEqual('foo'.upper(), 'FOO')

    def test_renyi(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_ab(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_beta(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_lp(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)
