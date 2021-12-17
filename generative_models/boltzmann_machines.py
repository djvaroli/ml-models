"""
Implementation of Restricted Boltzmann Machine and associated functions.
"""
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


class BinaryRestrictedBoltzmannMachine(nn.Module):
    """
    Implementation of the restricted Boltzmann Machine.
    See https://christian-igel.github.io/paper/TRBMAI.pdf for additional info
    """

    def __init__(
            self,
            n_visible_nodes: int,
            n_hidden_nodes: int,
            mcmc_steps: int = 10,
            weights_initialization_function: Callable = torch.randn,
            bias_initialization_function: Callable = torch.zeros
    ):
        super(BinaryRestrictedBoltzmannMachine, self).__init__()
        self.n_visible_nodes = n_visible_nodes
        self.n_hidden_nodes = n_hidden_nodes
        self.mcmc_steps = mcmc_steps
        self._weights_init_f = weights_initialization_function
        self._bias_init_f = bias_initialization_function
        self.weights = nn.Parameter(self._weights_init_f(n_visible_nodes, n_hidden_nodes) * 1e-2)
        self.visible_bias = nn.Parameter(self._bias_init_f(n_visible_nodes))
        self.hidden_bias = nn.Parameter(self._bias_init_f(n_hidden_nodes))

    @staticmethod
    def bernoulli_sample(probability_distribution: torch.Tensor):
        """
        Samples from a bernoulli distribution given a tensor corresponding to a probability distribution
        :param probability_distribution:
        :return:
        """
        return torch.bernoulli(probability_distribution)

    def sample_visible_given_hidden(self, observed_hidden: torch.Tensor) -> torch.Tensor:
        """
        Given observed hidden node values, compute the activation probabilities of the visible nodes and
        from the observed activations compute a sample of the visible units.
        :param observed_hidden: A Tensor of observed values in the hidden nodes of the RBM. This tensor will have
        dimension equal to the number of hidden nodes in the RBM.
        :return:
        """
        # linear will perform x \dot A.T + b
        visible_activations = torch.sigmoid(F.linear(observed_hidden, self.weights, self.visible_bias))

        return self.bernoulli_sample(visible_activations)

    def sample_hidden_given_visible(self, observed_visible: torch.Tensor) -> torch.Tensor:
        """
        Given observed visible node values, compute the activation probabilities of the hidden nodes and
        from the observed activations compute a sample of the hidden units.
        :param observed_visible: A Tensor of observed values in the visible nodes of the RBM. This tensor will
        have a dimension equal to the number of visible nodes.
        :return:
        """
        hidden_activations = torch.sigmoid(F.linear(observed_visible, self.weights.T, self.hidden_bias))
        return self.bernoulli_sample(hidden_activations)

    def free_energy(self, observed_visible: torch.Tensor) -> torch.Tensor:
        """
        Calculates the free energy of an observation.
        Refer to this to see an explanation of why code below is valid for a Restricted Boltzmann Machine and
        a brief derivation
        https://stats.stackexchange.com/questions/114844/how-to-compute-the-free-energy-of-a-rbm-given-its-energy

        E(v) = -vbias_T.v - sum(log(1 + exp(hbias + W_T.v)))

        :param observed_visible:
        :return:
        """
        wx_b = F.linear(input=observed_visible, weight=self.weights.T, bias=self.hidden_bias)
        aT_dot_v = observed_visible.mv(self.visible_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)

        return (-hidden_term - aT_dot_v).mean()

    def forward(self, observed_visible: torch.Tensor):
        """
        Carries out the forward step of the RBM
        :param observed_visible:
        :return:
        """
        sampled_hidden = self.sample_hidden_given_visible(observed_visible)
        sampled_visible = torch.empty()
        for _ in range(max(self.mcmc_steps, 1)):
            sampled_visible = self.sample_visible_given_hidden(sampled_hidden)
            sampled_hidden = self.sample_hidden_given_visible(sampled_visible)

        return observed_visible, sampled_visible

    def test(self, sample_input: torch.Tensor = None, batch_size: int = 10):
        """
        Performs basic tests of the network's functions to ensure no errors occur.
        :param: batch_size:
        :return:
        """
        if sample_input is None:
            sample_input = torch.empty(batch_size, self.n_visible_nodes).uniform_(0, 1)

        hidden = self.sample_hidden_given_visible(sample_input)
        visible = self.sample_visible_given_hidden(hidden)
        energy = self.free_energy(sample_input)
