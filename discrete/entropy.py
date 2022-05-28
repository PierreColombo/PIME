


from discrete_estimator import DiscreteEstimator




class DiscreteEntropyEstimator(DiscreteEstimator):
    def __init__(self, name, discret_estimator, **kwargs):
        self.name = name
        self.discret_estimator = discret_estimator(name, **kwargs)

    def predict(self, X, Y=None):
        """
        :param X: discreate input reference distribution over the vocabulary
        :param Y: discreate hypothesis reference distribution over the vocabulary
        :param alpha: alpha parameter of the divergence
        :return: alpha divergence between the reference and hypothesis distribution
        """
        if Y is not None:
            raise NotImplemented
        else:
            batch_size = X.size(0)
            tensor_length = X.size(1)
            U = torch.tensor([1 / tensor_length] * tensor_length).unsqueeze(0).repeat(batch_size, 1).to(
                X.device)
            return self.discret_estimator.predict(X, U)
