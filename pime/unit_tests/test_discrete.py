import unittest
from tqdm import tqdm
import torch
from divergence import DISCRETE_ESTIMATORS


class TestDiscreteMethods(unittest.TestCase):

    def test_alpha(self):
        for alpha in tqdm([-2, -1.01, -0.99, -0.01, 0.01, 0.99, 1.01, 2], 'Alpha Div'):
            batch_size = 10
            tensor_length = 4
            uniform_tensor = torch.tensor([1 / tensor_length for _ in range(tensor_length)])
            batched_uniform_tensor = uniform_tensor.unsqueeze(0).repeat(batch_size, 1)
            random_vector = torch.nn.Softmax(dim=-1)(torch.rand(batch_size, tensor_length))
            alpha_div = DISCRETE_ESTIMATORS['alpha_div']('test_alpha_div', alpha)
            alpha_entropy = DISCRETE_ESTIMATORS['entropy']('test_alpha_entropy', DISCRETE_ESTIMATORS['alpha_div'], alpha=alpha)
            assert torch.sum(
                torch.isclose(alpha_div.predict(random_vector, random_vector), torch.zeros(batch_size), atol=1e-07))
            alpha_div_value = alpha_div.predict(random_vector, batched_uniform_tensor)
            alpha_entropy_value = alpha_entropy.predict(random_vector, None)
            assert torch.sum(torch.isclose(alpha_div_value, alpha_entropy_value, atol=1e-07))
            assert torch.sum(
                torch.isclose(alpha_entropy.predict(batched_uniform_tensor), torch.zeros(batch_size), atol=1e-07))

    def test_beta(self):
        for beta in tqdm([-2, -1.01, -0.99, -0.01, 0.01, 0.99, 1.01, 2], 'Beta Div'):
            batch_size = 10
            tensor_length = 4
            uniform_tensor = torch.tensor([1 / tensor_length for _ in range(tensor_length)])
            batched_uniform_tensor = uniform_tensor.unsqueeze(0).repeat(batch_size, 1)
            random_vector = torch.nn.Softmax(dim=-1)(torch.rand(batch_size, tensor_length))
            beta_div = DISCRETE_ESTIMATORS['beta_div']('test_beta_div', beta)
            beta_entropy = DISCRETE_ESTIMATORS['entropy']('test_beta_entropy', DISCRETE_ESTIMATORS['beta_div'], beta=beta)
            print( beta_div.predict(random_vector, random_vector))
            assert torch.sum(
                torch.isclose(beta_div.predict(random_vector, random_vector), torch.zeros(batch_size), atol=1e-05))
            beta_div_value = beta_div.predict(random_vector, batched_uniform_tensor)
            beta_entropy_value = beta_entropy.predict(random_vector, None)
            assert torch.sum(torch.isclose(beta_div_value, beta_entropy_value, atol=1e-05))
            assert torch.sum(
                torch.isclose(beta_entropy.predict(batched_uniform_tensor), torch.zeros(batch_size), atol=1e-05))

    def test_renyi(self):
        for alpha in tqdm([-2, -1.01, -0.99, -0.01, 0.01, 0.99, 1.01, 2], 'Renyi Div'):
            batch_size = 10
            tensor_length = 4
            uniform_tensor = torch.tensor([1 / tensor_length for _ in range(tensor_length)])
            batched_uniform_tensor = uniform_tensor.unsqueeze(0).repeat(batch_size, 1)
            random_vector = torch.nn.Softmax(dim=-1)(torch.rand(batch_size, tensor_length))
            renyi_div = DISCRETE_ESTIMATORS['renyi_div']('test_renyi_div', alpha)
            renyi_entropy = DISCRETE_ESTIMATORS['entropy']('test_renyi_entropy', DISCRETE_ESTIMATORS['renyi_div'], alpha=alpha)
            renyi_div_value = renyi_div.predict(random_vector, batched_uniform_tensor)
            renyi_entropy_value = renyi_entropy.predict(random_vector, None)
            assert torch.sum(
                torch.isclose(renyi_div.predict(random_vector, random_vector), torch.zeros(batch_size), atol=1e-07))
            assert torch.sum(torch.isclose(renyi_div_value, renyi_entropy_value, atol=1e-07))
            assert torch.sum(
                torch.isclose(renyi_entropy.predict(batched_uniform_tensor), torch.zeros(batch_size), atol=1e-07))

    def test_ab(self):
        for alpha in tqdm([0.2, 0.677, 2], 'AB Div'):
            for beta in tqdm([0.5], 'AB Div'):
                if alpha + beta != 0:
                    batch_size = 10
                    tensor_length = 4
                    uniform_tensor = torch.tensor([1 / tensor_length for _ in range(tensor_length)])
                    batched_uniform_tensor = uniform_tensor.unsqueeze(0).repeat(batch_size, 1)
                    random_vector = torch.nn.Softmax(dim=-1)(torch.rand(batch_size, tensor_length))
                    ab_div = DISCRETE_ESTIMATORS['ab_div']('test_ab_div', alpha=alpha, beta=beta)
                    ab_entropy = DISCRETE_ESTIMATORS['entropy']('test_ab_entropy', DISCRETE_ESTIMATORS['ab_div'], alpha=alpha, beta=beta)
                    ab_div_value = ab_div.predict(random_vector, batched_uniform_tensor)
                    ab_entropy_value = ab_entropy.predict(random_vector, None)
                    assert torch.sum(
                        torch.isclose(ab_div.predict(random_vector, random_vector), torch.zeros(batch_size),
                                      atol=1e-7))
                    assert torch.sum(torch.isclose(ab_div_value, ab_entropy_value, atol=1e-7))
                    assert torch.sum(torch.isclose(ab_entropy.predict(batched_uniform_tensor), torch.zeros(batch_size),
                                                   atol=1e-7))

    def test_lp(self):
        for p in tqdm([-1, 3, 4], 'Renyi Div'):
            batch_size = 10
            tensor_length = 4
            uniform_tensor = torch.tensor([1 / tensor_length for _ in range(tensor_length)])
            batched_uniform_tensor = uniform_tensor.unsqueeze(0).repeat(batch_size, 1)
            random_vector = torch.nn.Softmax(dim=-1)(torch.rand(batch_size, tensor_length))
            lp = DISCRETE_ESTIMATORS['lp']('test_lp', p)
            lp_uni = DISCRETE_ESTIMATORS['entropy']('test_lp_uni', DISCRETE_ESTIMATORS['lp'], power=p)
            lp_value = lp.predict(random_vector, batched_uniform_tensor)
            lp_uni_value = lp_uni.predict(random_vector, None)
            assert torch.sum(
                torch.isclose(lp.predict(random_vector, random_vector), torch.zeros(batch_size), atol=1e-07))
            assert torch.sum(torch.isclose(lp_value, lp_uni_value, atol=1e-07))
            assert torch.sum(
                torch.isclose(lp_uni.predict(batched_uniform_tensor), torch.zeros(batch_size), atol=1e-07))

    def test_fisher_rao(self):
        batch_size = 10
        tensor_length = 4
        uniform_tensor = torch.tensor([1 / tensor_length for _ in range(tensor_length)])
        batched_uniform_tensor = uniform_tensor.unsqueeze(0).repeat(batch_size, 1)
        random_vector = torch.nn.Softmax(dim=-1)(torch.rand(batch_size, tensor_length))
        fisher_rao = DISCRETE_ESTIMATORS['fisher_rao']('test_fr')
        fisher_rao_uni = DISCRETE_ESTIMATORS['entropy']('test_fr_uni', DISCRETE_ESTIMATORS['fisher_rao'])
        fisher_rao_value = fisher_rao.predict(random_vector, batched_uniform_tensor)
        fisher_rao_uni_value = fisher_rao_uni.predict(random_vector, None)
        assert torch.sum(
            torch.isclose(fisher_rao.predict(random_vector, random_vector), torch.zeros(batch_size), atol=1e-07))
        assert torch.sum(torch.isclose(fisher_rao_value, fisher_rao_uni_value, atol=1e-07))
        assert torch.sum(
            torch.isclose(fisher_rao_uni.predict(batched_uniform_tensor), torch.zeros(batch_size), atol=1e-07))

    def test_kl(self):
        batch_size = 10
        tensor_length = 4
        uniform_tensor = torch.tensor([1 / tensor_length for _ in range(tensor_length)])
        batched_uniform_tensor = uniform_tensor.unsqueeze(0).repeat(batch_size, 1)
        random_vector = torch.nn.Softmax(dim=-1)(torch.rand(batch_size, tensor_length))
        kl = DISCRETE_ESTIMATORS['kl_div']('test_kl_div')
        shanon_entropy = DISCRETE_ESTIMATORS['entropy']('test_shannon_entropy', DISCRETE_ESTIMATORS['kl_div'])
        kl_value = kl.predict(random_vector, batched_uniform_tensor)
        shanon_entropy_value = shanon_entropy.predict(random_vector, None)
        assert torch.sum(
            torch.isclose(kl.predict(random_vector, random_vector), torch.zeros(batch_size), atol=1e-07))
        assert torch.sum(torch.isclose(kl_value, shanon_entropy_value, atol=1e-07))
        assert torch.sum(
            torch.isclose(shanon_entropy.predict(batched_uniform_tensor), torch.zeros(batch_size), atol=1e-07))
