import torch

from cardiogb.losses.distribution import rbf_mmd


def _dense_rbf_mmd(predicted, observed, bandwidths):
    xx = torch.cdist(predicted, predicted).square()
    yy = torch.cdist(observed, observed).square()
    xy = torch.cdist(predicted, observed).square()
    values = []
    for bandwidth in bandwidths:
        denominator = 2 * bandwidth**2
        values.append(
            torch.exp(-xx / denominator).mean()
            + torch.exp(-yy / denominator).mean()
            - 2 * torch.exp(-xy / denominator).mean()
        )
    return torch.stack(values).mean()


def test_checkpointed_chunked_mmd_matches_dense_value_and_gradient():
    generator = torch.Generator().manual_seed(20260823)
    predicted = torch.randn(7, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    observed = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    bandwidths = (0.1, 0.5, 1.0, 2.0)

    chunked = rbf_mmd(predicted, observed, bandwidths, chunk_size=2)
    chunked_gradient = torch.autograd.grad(chunked, predicted, retain_graph=True)[0]
    dense = _dense_rbf_mmd(predicted, observed, bandwidths)
    dense_gradient = torch.autograd.grad(dense, predicted)[0]

    assert torch.allclose(chunked, dense, rtol=1e-12, atol=1e-12)
    assert torch.allclose(chunked_gradient, dense_gradient, rtol=1e-11, atol=1e-12)
