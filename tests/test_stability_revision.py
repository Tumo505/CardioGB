import torch

from cardiogb.losses import LossWeights
from cardiogb.losses.objective import cardiogb_objective
from cardiogb.models import GraphNeuralODEFunc, MechanisticODE
from cardiogb.models.cardiogb import CardioGB
from cardiogb.models.message_passing import TorchGraph
from cardiogb.ode.solvers import integrate_fixed_step
from cardiogb.training import CrossSectionalTrainer, CrossSectionalTransition, TrainerConfig
from cardiogb.utils.config import load_yaml


def graph(nodes: int) -> TorchGraph:
    source = torch.arange(nodes - 1)
    target = source + 1
    edges = torch.cat((torch.stack((source, target)), torch.stack((target, source))), dim=1)
    return TorchGraph(edges, torch.zeros(edges.shape[1], 3))


def test_projected_solver_preserves_closed_state_interval() -> None:
    initial = torch.tensor([[0.9]], requires_grad=True)
    result = integrate_fixed_step(
        lambda _t, x: 100 * torch.ones_like(x),
        initial,
        0.0,
        10.0,
        step_size=0.5,
        projector=lambda x: x.clamp(0, 1),
    )
    assert torch.isfinite(result).all()
    assert torch.all((0 <= result) & (result <= 1))


def test_cardiogb_long_horizon_is_bounded_and_residual_is_scaled() -> None:
    mechanistic = MechanisticODE.from_config(load_yaml("configs/mechanistic_model.yaml"))
    residual = GraphNeuralODEFunc(hidden_dim=8, layers=1)
    model = CardioGB(
        mechanistic,
        residual,
        residual_scale_max=0.2,
        residual_scale_initial=0.02,
    )
    states = torch.rand(12, 6)
    prediction = model.integrate(states, graph(12), 3.0, 28.0, step_size=0.25)
    assert torch.isfinite(prediction).all()
    assert torch.all((0 <= prediction) & (prediction <= 1))
    assert torch.all(model.residual_scale() <= 0.2)
    components = model.vector_field(3.0, states, graph(12))
    assert components["residual"].abs().max() <= 0.2


def test_disjoint_patch_batch_offsets_graph_edges_without_duplicating_targets() -> None:
    target = torch.rand(9, 6)
    first = CrossSectionalTransition(torch.rand(3, 6), target, graph(3), 0, 1, "a", "0_to_1")
    second = CrossSectionalTransition(torch.rand(4, 6), target.clone(), graph(4), 0, 1, "b", "0_to_1")
    combined = CrossSectionalTrainer._combine_transition_patches([first, second])
    assert combined.source_states.shape == (7, 6)
    assert combined.target_states.shape == target.shape
    assert combined.graph.edge_index.max().item() == 6
    edge_set = set(map(tuple, combined.graph.edge_index.T.tolist()))
    assert (2, 3) not in edge_set and (3, 2) not in edge_set


def test_augmented_distribution_objective_is_finite_and_differentiable() -> None:
    predicted = torch.rand(32, 6, requires_grad=True)
    observed = torch.rand(41, 6)
    total, components = cardiogb_objective(
        predicted,
        observed,
        weights=LossWeights(distribution=1, moments=0.25, wasserstein=0.25),
    )
    assert {"distribution", "moments", "wasserstein"} <= components.keys()
    assert torch.isfinite(total)
    total.backward()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()
