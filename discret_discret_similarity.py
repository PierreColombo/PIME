from __future__ import absolute_import, division, print_function
import torch

from abstract_class import DiscreteEstimator


class AlphaDivergence(DiscreteEstimator):
    def __init__(self, name, alpha):
        self.name = name
        self.alpha = alpha

    def predict(self, X, Y):
        """
        :param X: discreate input reference distribution over the vocabulary
        :param Y: discreate hypothesis reference distribution over the vocabulary
        :param alpha: alpha parameter of the divergence
        :return: alpha divergence between the reference and hypothesis distribution
        """
        alpha = self.alpha
        assert alpha != 1 and alpha != 0
        return 1 / (alpha * (alpha - 1)) - torch.sum(X ** alpha * Y ** (1 - alpha), dim=-1) / (
                alpha * (alpha - 1))


class LP(DiscreteEstimator):
    def __init__(self, name, power):
        self.name = name
        self.alpha = power

    def predict(self, X, Y):
        """
        :param ref_dist: discreate input reference distribution over the vocabulary
        :param hypo_dist: discreate hypothesis reference distribution over the vocabulary
        :return: l1 norm between the reference and hypothesis distribution
        """
        return torch.norm(X - Y, p=self.power, dim=-1)


class FisherRao(DiscreteEstimator):
    def __init__(self, name):
        self.name = name

    def predict(self, X, Y):
        """
        :param ref_dist: discreate input reference distribution over the vocabulary
        :param hypo_dist: discreate hypothesis reference distribution over the vocabulary
        :return: fisher rao distance between the reference and hypothesis distribution
        """
        fisher_rao = torch.clamp(
            torch.sum(torch.sqrt(X) * torch.sqrt(
                Y),
                      dim=-1), 0, 1)
        return 2 * torch.acos(fisher_rao)


class KullbackLeiblerDivergence(DiscreteEstimator):
    def __init__(self, name):
        self.name = name

    def predict(self, X, Y):
        """
        :param ref_dist: discreate input reference distribution over the vocabulary
        :param hypo_dist: discreate hypothesis reference distribution over the vocabulary
        :return: fisher rao distance between the reference and hypothesis distribution
        """
        fisher_rao = torch.clamp(
            torch.sum(torch.sqrt(X) * torch.sqrt(
                Y),
                      dim=-1), 0, 1)
        return 2 * torch.acos(fisher_rao)


class RenyiDivergence(DiscreteEstimator):
    def __init__(self, name, alpha):
        self.name = name
        self.alpha = alpha
        assert self.alpha != 1

    def predict(self, X, Y):
        """
        :param ref_dist: discreate input reference distribution over the vocabulary
        :param hypo_dist: discreate hypothesis reference distribution over the vocabulary
        :return: fisher rao distance between the reference and hypothesis distribution
        """

        return torch.log(torch.sum(X ** self.alpha * Y ** (1 - self.alpha), dim=-1)) / (self.alpha - 1)


class BethaDivergence(DiscreteEstimator):
    def __init__(self, name, beta):
        self.name = name
        self.beta = beta
        assert self.beta != -1
        assert self.beta != 0

    def predict(self, X, Y):
        """
        :param ref_dist: discreate input reference distribution over the vocabulary
        :param hypo_dist: discreate hypothesis reference distribution over the vocabulary
        :return: beta divergence between the reference and hypothesis distribution
        """

        first_term = torch.log(torch.sum(X ** (self.beta + 1), dim=-1)) / (self.beta * (self.beta + 1))
        second_term = torch.log(torch.sum(Y ** (self.beta + 1), dim=-1)) / (self.beta + 1)
        third_term = torch.log(torch.sum(X * Y ** (self.beta), dim=-1)) / (self.beta)
        return first_term + second_term - third_term


class ABDivergence(DiscreteEstimator):
    def __init__(self, name, alpha, beta):
        self.name = name
        self.alpha = alpha
        self.beta = beta
        assert self.alpha != 0
        assert self.beta != 0
        assert self.beta + self.alpha != 0

    def predict(self, X, Y):
        """
        :param ref_dist: discreate input reference distribution over the vocabulary
        :param hypo_dist: discreate hypothesis reference distribution over the vocabulary
        :return:  ab divergence between the reference and hypothesis distribution
        """

        first_term = torch.log(torch.sum(X ** (self.beta + self.alpha), dim=-1)) / (
                self.beta * (self.beta + self.alpha))
        second_term = torch.log(torch.sum(Y ** (self.beta + self.alpha), dim=-1)) / (
                self.alpha * (self.beta + self.alpha))
        third_term = torch.log(torch.sum((X ** (self.alpha)) * (Y ** (self.beta)), dim=-1)) / (self.beta * self.alpha)
        return first_term + second_term - third_term
