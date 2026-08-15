import torch

from cardiogb.interpretation import mechanistic_insufficiency


def test_mechanistic_insufficiency_limits() -> None:
    mech = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    residual = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    score = mechanistic_insufficiency(mech, residual)
    assert torch.allclose(score, torch.tensor([0.0, 1.0]))
