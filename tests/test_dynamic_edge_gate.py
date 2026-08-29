from __future__ import annotations

import torch

from cardiogb.models.graph_neural_ode import GraphNeuralODEFunc
from cardiogb.models.message_passing import TorchGraph


def test_state_dependent_edge_gate_is_bounded_and_trainable() -> None:
    model = GraphNeuralODEFunc(
        state_dim=6,
        edge_dim=3,
        hidden_dim=12,
        layers=2,
        dropout=0.0,
        edge_gating=True,
    )
    states = torch.rand(5, 6, requires_grad=True)
    graph = TorchGraph(
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
        edge_attr=torch.rand(5, 3),
    )
    edge_index = graph.edge_index
    source, target = edge_index
    features = torch.cat((
        torch.cat((states, torch.zeros(5, 1)), dim=-1)[target],
        torch.cat((states, torch.zeros(5, 1)), dim=-1)[source],
        graph.edge_attr,
    ), dim=-1)
    gates = torch.sigmoid(model.message_passing.edge_gate_network(features))
    assert torch.all((gates > 0) & (gates < 1))
    assert float(gates.mean().detach()) > 0.75
    output = model(0.0, states, graph)
    output.square().mean().backward()
    gradients = [parameter.grad for parameter in model.message_passing.edge_gate_network.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_edgeless_external_graph_remains_finite_with_edge_gating() -> None:
    model = GraphNeuralODEFunc(edge_gating=True)
    states = torch.rand(3, 6)
    graph = TorchGraph(torch.empty((2, 0), dtype=torch.long), torch.empty((0, 3)))
    output = model(0.0, states, graph)
    assert output.shape == states.shape
    assert torch.isfinite(output).all()
