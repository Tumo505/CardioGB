import torch

from cardiogb.models.mechanistic import MechanisticODE
from cardiogb.utils.config import load_yaml


def test_mechanistic_model_has_positive_rates_and_expected_shape() -> None:
    model = MechanisticODE.from_config(load_yaml("configs/mechanistic_model.yaml"))
    states = torch.zeros(4, 6)
    derivative = model(torch.tensor(0.0), states)
    assert derivative.shape == states.shape
    assert derivative[:, 1].min() > 0  # early injury input activates A
    assert all(value.item() > 0 for value in model.constrained_parameters().values())


def test_mechanistic_model_propagates_gradients() -> None:
    model = MechanisticODE.from_config(load_yaml("configs/mechanistic_model.yaml"))
    loss = model(0.5, torch.rand(3, 6)).square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())

