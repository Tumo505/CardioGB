import torch

from cardiogb.models import GraphNeuralODEFunc, MechanisticODE
from cardiogb.models.cardiogb import CardioGB
from cardiogb.models.message_passing import TorchGraph
from cardiogb.utils.config import load_yaml


def test_mechanistic_gate_is_bounded_and_keeps_vector_field_contract() -> None:
    model = CardioGB(
        MechanisticODE.from_config(load_yaml("configs/mechanistic_model.yaml")),
        GraphNeuralODEFunc(hidden_dim=8, layers=1),
        residual_scale_max=1.0,
        residual_scale_initial=0.1,
        mechanistic_gate_min=0.05,
        mechanistic_gate_initial=0.5,
    )
    nodes = 8
    source = torch.arange(nodes - 1)
    target = source + 1
    edges = torch.cat((torch.stack((source, target)), torch.stack((target, source))), dim=1)
    graph = TorchGraph(edges, torch.zeros(edges.shape[1], 3))
    components = model.vector_field(0.0, torch.rand(nodes, 6), graph)
    assert set(components) == {"total", "mechanistic", "residual"}
    assert torch.all((model.mechanistic_gate() >= 0.05) & (model.mechanistic_gate() <= 1.0))
    assert torch.all((model.residual_scale() > 0) & (model.residual_scale() <= 1.0))
