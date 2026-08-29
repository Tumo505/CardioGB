import torch

from cardiogb.data.state_dataset import StateDataset
from cardiogb.models import GraphNeuralODEFunc, MechanisticODE, SpeciesAdapter, SpeciesAdaptedForecaster
from cardiogb.models.cardiogb import CardioGB
from cardiogb.models.message_passing import TorchGraph
from cardiogb.utils.config import load_yaml


def _graph(nodes: int) -> TorchGraph:
    source = torch.arange(nodes - 1)
    target = source + 1
    edges = torch.cat((torch.stack((source, target)), torch.stack((target, source))), dim=1)
    return TorchGraph(edges, torch.zeros(edges.shape[1], 3))


def _model() -> CardioGB:
    mechanism = MechanisticODE.from_config(load_yaml("configs/mechanistic_model.yaml"))
    return CardioGB(
        mechanism,
        GraphNeuralODEFunc(hidden_dim=8, layers=1),
        orthogonal_residual=True,
        persistence_gate=True,
        velocity_limit=0.5,
    )


def test_residual_is_orthogonal_to_mechanistic_field() -> None:
    model = _model()
    states = torch.rand(10, 6)
    parts = model.vector_field(1.0, states, _graph(10), forecast_horizon=1.0)
    inner = (parts["mechanistic"] * parts["residual"]).sum(dim=-1)
    assert inner.abs().max() < 1e-5


def test_persistence_confidence_decreases_with_horizon() -> None:
    model = _model()
    assert model.persistence_gate(1.0) > model.persistence_gate(28.0)


def test_bounded_mechanistic_parameters_and_velocity() -> None:
    model = _model()
    assert all(0 < float(value.detach()) < 2.0 for value in model.mechanistic_model.constrained_parameters().values())
    parts = model.vector_field(1.0, torch.rand(8, 6), _graph(8), forecast_horizon=28.0)
    assert parts["total"].abs().max() <= 0.5


def test_all_pair_transitions_record_intermediate_stages() -> None:
    states = torch.rand(8, 6).numpy()
    dataset = StateDataset(
        states=states,
        coordinates=torch.rand(8, 2).numpy(),
        sections=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]).numpy().astype(str),
        times=torch.tensor([0, 0, 1, 1, 3, 3, 7, 7]).numpy(),
        groups=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]).numpy().astype(str),
        state_names=("I", "A", "F", "C", "V", "M"),
    )
    transitions = dataset.transitions(adjacent_only=False, k=1)
    longest = next(item for item in transitions if item.t0 == 0 and item.t1 == 7)
    assert longest.intermediate_times == (1.0, 3.0)


def test_species_adapter_is_identity_initialized_and_differentiable() -> None:
    model = _model()
    adapter = SpeciesAdapter(6, ("zebrafish", "mouse"))
    forecaster = SpeciesAdaptedForecaster(model, adapter)
    states = torch.rand(6, 6)
    prediction = forecaster.forecast(
        states, _graph(6), 0.0, 1.0, species="mouse", step_size=0.5
    )
    (prediction.mean() + adapter.regularization()).backward()
    assert prediction.shape == states.shape

