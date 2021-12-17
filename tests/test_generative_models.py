import generative_models
from generative_models import boltzmann_machines


def test_version():
    assert generative_models.__version__ == "0.1.0"


def test_binary_restricted_boltzmann_machine():
    """
    Tests that the basic operations defined in the BinaryRestrictedBoltzmannMachine work without
    raising any obvious errors.
    :return:
    """

    rbm = boltzmann_machines.BinaryRestrictedBoltzmannMachine(10, 10)
    assert rbm.test() is None
