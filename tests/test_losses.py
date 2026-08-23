import torch

from cardiogb.losses import LossWeights, cardiogb_objective, rbf_mmd
from cardiogb.models.message_passing import TorchGraph


def test_mmd_is_small_for_identical_samples_and_differentiable() -> None:
    sample = torch.rand(20, 6, requires_grad=True)
    loss = rbf_mmd(sample, sample)
    assert abs(float(loss.detach())) < 1e-6
    loss.backward()
    assert sample.grad is not None


def test_composite_objective_accepts_unmatched_counts() -> None:
    predicted = torch.rand(12, 6, requires_grad=True)
    observed = torch.rand(17, 6)
    graph = TorchGraph(torch.tensor([[0, 1], [1, 0]]), torch.tensor([[1., 1., 0.], [1., -1., 0.]]))
    loss, parts = cardiogb_objective(
        predicted,
        observed,
        graph=graph,
        residual=torch.rand_like(predicted),
        weights=LossWeights(1.0, 0.1, 0.1, 0.1, 0.1),
    )
    loss.backward()
    assert set(parts) == {
        "distribution",
        "moments",
        "wasserstein",
        "biology",
        "spatial",
        "residual",
    }
