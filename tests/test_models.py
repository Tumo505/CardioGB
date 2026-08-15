import torch

from cardiogb.models import CardioGB, GraphNeuralODEFunc, MechanisticODE, NeuralODEFunc
from cardiogb.models.message_passing import TorchGraph
from cardiogb.utils.config import load_yaml


def toy_graph() -> TorchGraph:
    return TorchGraph(
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.tensor([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0]] * 2),
    )


def test_neural_and_graph_vector_fields() -> None:
    states = torch.rand(3, 6)
    assert NeuralODEFunc()(0.5, states).shape == states.shape
    assert GraphNeuralODEFunc()(0.5, states, toy_graph()).shape == states.shape


def test_cardiogb_exposes_components_and_integrates() -> None:
    states = torch.rand(3, 6)
    graph = toy_graph()
    mechanistic = MechanisticODE.from_config(load_yaml("configs/mechanistic_model.yaml"))
    residual = GraphNeuralODEFunc()
    model = CardioGB(mechanistic, residual)
    components = model.vector_field(0.0, states, graph)
    assert set(components) == {"total", "mechanistic", "residual"}
    assert torch.allclose(components["total"], components["mechanistic"] + components["residual"])
    prediction = model.integrate(states, graph, 0.0, 0.2, step_size=0.1)
    assert prediction.shape == states.shape
