import math

import torch

from cardiogb.ode.solvers import integrate_fixed_step


def test_rk4_integrates_exponential_decay() -> None:
    x0 = torch.tensor([1.0], requires_grad=True)
    result = integrate_fixed_step(lambda _t, x: -x, x0, 0.0, 1.0, step_size=0.05, method="rk4")
    assert torch.allclose(result, torch.tensor([math.exp(-1)]), atol=1e-5)
    result.sum().backward()
    assert x0.grad is not None

