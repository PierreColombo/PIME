import torch


class Divergence:

    def predict(self, X, Y):
        raise NotImplementedError


class AlphaDivergence(Divergence):
    def __init__(self, name, alpha):
        self.name = name
        self.alpha = alpha
        assert alpha != 1 and alpha != 0

    def predict(self, X, Y):
        """
        :param X: discrete input reference distribution over the vocabulary
        :param Y: discrete hypothesis reference distribution over the vocabulary
        :param alpha: alpha parameter of the divergence
        :return: alpha divergence between the reference and hypothesis distribution
        """
        alpha = self.alpha

        return 1 / (alpha * (alpha - 1)) - torch.sum(X ** alpha * Y ** (1 - alpha), dim=-1) / (
                alpha * (alpha - 1))


class LP(Divergence):
    def __init__(self, name, power):
        self.name = name
        self.alpha = power

    def predict(self, X, Y):
        """
        :param ref_dist: discrete input reference distribution over the vocabulary
        :param hypo_dist: discrete hypothesis reference distribution over the vocabulary
        :return: l1 norm between the reference and hypothesis distribution
        """
        return torch.norm(X - Y, p=self.power, dim=-1)


class FisherRao(Divergence):
    def __init__(self, name):
        self.name = name

    def predict(self, X, Y):
        """
        :param ref_dist: discrete input reference distribution over the vocabulary
        :param hypo_dist: discrete hypothesis reference distribution over the vocabulary
        :return: fisher rao distance between the reference and hypothesis distribution
        """
        fisher_rao = torch.clamp(
            torch.sum(torch.sqrt(X) * torch.sqrt(
                Y),
                      dim=-1), 0, 1)
        return 2 * torch.acos(fisher_rao)


class KullbackLeiblerDivergence(Divergence):
    def __init__(self, name):
        self.name = name

    def predict(self, X, Y):
        """
        :param ref_dist: discrete input reference distribution over the vocabulary
        :param hypo_dist: discrete hypothesis reference distribution over the vocabulary
        :return: fisher rao distance between the reference and hypothesis distribution
        """
        fisher_rao = torch.clamp(
            torch.sum(torch.sqrt(X) * torch.sqrt(
                Y),
                      dim=-1), 0, 1)
        return 2 * torch.acos(fisher_rao)


class RenyiDivergence(Divergence):
    def __init__(self, name, alpha):
        self.name = name
        self.alpha = alpha
        assert self.alpha != 1

    def predict(self, X, Y):
        """
        :param ref_dist: discrete input reference distribution over the vocabulary
        :param hypo_dist: discrete hypothesis reference distribution over the vocabulary
        :return: fisher rao distance between the reference and hypothesis distribution
        """

        return torch.log(torch.sum(X ** self.alpha * Y ** (1 - self.alpha), dim=-1)) / (self.alpha - 1)


class BetaDivergence(Divergence):
    def __init__(self, name, beta):
        self.name = name
        self.beta = beta
        assert self.beta != -1
        assert self.beta != 0

    def predict(self, X, Y):
        """
        :param ref_dist: discrete input reference distribution over the vocabulary
        :param hypo_dist: discrete hypothesis reference distribution over the vocabulary
        :return: beta divergence between the reference and hypothesis distribution
        """

        first_term = torch.log(torch.sum(X ** (self.beta + 1), dim=-1)) / (self.beta * (self.beta + 1))
        second_term = torch.log(torch.sum(Y ** (self.beta + 1), dim=-1)) / (self.beta + 1)
        third_term = torch.log(torch.sum(X * Y ** (self.beta), dim=-1)) / (self.beta)
        return first_term + second_term - third_term


class ABDivergence(Divergence):
    def __init__(self, name, alpha, beta):
        self.name = name
        self.alpha = alpha
        self.beta = beta
        assert self.alpha != 0
        assert self.beta != 0
        assert self.beta + self.alpha != 0

    def predict(self, X, Y):
        """
        :param ref_dist: discrete input reference distribution over the vocabulary
        :param hypo_dist: discrete hypothesis reference distribution over the vocabulary
        :return:  ab divergence between the reference and hypothesis distribution
        """

        first_term = torch.log(torch.sum(X ** (self.beta + self.alpha), dim=-1)) / (
                self.beta * (self.beta + self.alpha))
        second_term = torch.log(torch.sum(Y ** (self.beta + self.alpha), dim=-1)) / (
                self.alpha * (self.beta + self.alpha))
        third_term = torch.log(torch.sum((X ** (self.alpha)) * (Y ** (self.beta)), dim=-1)) / (self.beta * self.alpha)
        return first_term + second_term - third_term


class JeffreySymmetricDivergence(Divergence):
    def __init__(self, name, divergence, **kwargs):
        self.name = name
        self.divergence = divergence(name, **kwargs)

    def predict(self, X, Y=None):
        """
        :param X: discrete input reference distribution over the vocabulary
        :param Y: discrete hypothesis reference distribution over the vocabulary
        :param alpha: alpha parameter of the divergence
        :return: alpha divergence between the reference and hypothesis distribution
        """
        return (self.divergence.predict(X, Y) + self.divergence.predict(Y, X)) / 2


class JensenSymmetricDivergence(Divergence):
    def __init__(self, name, divergence, **kwargs):
        self.name = name
        self.divergence = divergence(name, **kwargs)

    def predict(self, X, Y=None):
        """
        :param X: discrete input reference distribution over the vocabulary
        :param Y: discrete hypothesis reference distribution over the vocabulary
        :param alpha: alpha parameter of the divergence
        :return: alpha divergence between the reference and hypothesis distribution
        """
        return (self.divergence.predict(Y, (X + Y) / 2) + self.divergence.predict(X, (X + Y) / 2)) / 2