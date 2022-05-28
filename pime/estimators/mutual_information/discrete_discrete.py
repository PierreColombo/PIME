from __future__ import absolute_import, division, print_function
import torch

from abstract_class import DiscreteEstimator







if __name__ == '__main__':
    from tqdm import tqdm

    batch_size = 10
    tensor_length = 4
    uniform_tensor = torch.tensor([1 / tensor_length for _ in range(tensor_length)])
    batched_uniform_tensor = uniform_tensor.unsqueeze(0).repeat(batch_size, 1)

    random_vector = torch.nn.Softmax(dim=-1)(torch.rand(batch_size, tensor_length))
    entropy = DiscreteEntropyEstimator('name', ABDivergence, alpha=1, beta=3)
    print(entropy.predict(random_vector, None))
    print(entropy.predict(batched_uniform_tensor, None))

    ab_div = ABDivergence('name', alpha=1, beta=3)
    print(ab_div.predict(random_vector, random_vector))
    print(ab_div.predict(batched_uniform_tensor, batched_uniform_tensor))

    for alpha in tqdm([0.2, 0.677, 2], 'AB Div'):
        for beta in tqdm([0.5], 'AB Div'):
            if alpha + beta != 0:
                batch_size = 10
                tensor_length = 4
                uniform_tensor = torch.tensor([1 / tensor_length for _ in range(tensor_length)])
                batched_uniform_tensor = uniform_tensor.unsqueeze(0).repeat(batch_size, 1)
                random_vector = torch.nn.Softmax(dim=-1)(torch.rand(batch_size, tensor_length))
                ab_div = ABDivergence('test_ab_div', alpha=alpha, beta=beta)
                ab_entropy = DiscreteEntropyEstimator('test_ab_entropy', ABDivergence, alpha=alpha, beta=beta)
                ab_div_value = ab_div.predict(random_vector, batched_uniform_tensor)
                ab_entropy_value = ab_entropy.predict(random_vector, None)
                assert torch.sum(torch.isclose(ab_div.predict(random_vector, random_vector), torch.zeros(batch_size),
                                               atol=1e-7))
                assert torch.sum(torch.isclose(ab_div_value, ab_entropy_value, atol=1e-7))
                assert torch.sum(torch.isclose(ab_entropy.predict(batched_uniform_tensor), torch.zeros(batch_size),
                                               atol=1e-7))
