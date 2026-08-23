import torch

from cardiogb.losses import LossWeights
from cardiogb.models import GraphNeuralODEFunc, MechanisticODE
from cardiogb.models.cardiogb import CardioGB
from cardiogb.models.message_passing import TorchGraph
from cardiogb.training import CrossSectionalTrainer, CrossSectionalTransition, TrainerConfig
from cardiogb.utils.config import load_yaml


def _graph(nodes: int) -> TorchGraph:
    source = torch.arange(nodes - 1)
    target = source + 1
    edges = torch.cat((torch.stack((source, target)), torch.stack((target, source))), dim=1)
    return TorchGraph(edges, torch.zeros(edges.shape[1], 3))


def test_cardiogb_mechanistic_warm_start_then_joint_training() -> None:
    model = CardioGB(
        MechanisticODE.from_config(load_yaml("configs/mechanistic_model.yaml")),
        GraphNeuralODEFunc(hidden_dim=8, layers=1),
    )
    transition = CrossSectionalTransition(
        torch.rand(10, 6), torch.rand(13, 6), _graph(10), 0.0, 0.1
    )
    trainer = CrossSectionalTrainer(
        model,
        device="cpu",
        config=TrainerConfig(
            epochs=1,
            warm_start_epochs=1,
            step_size=0.1,
            max_ode_steps_per_transition=2,
            gradient_checkpointing=False,
        ),
        loss_weights=LossWeights(distribution=1.0, residual=0.1),
    )
    history = trainer.fit([transition], [transition])
    assert [row["warm_start"] for row in history] == [1.0, 0.0]
    assert model.residual_enabled
    assert all(parameter.requires_grad for parameter in model.residual_model.parameters())
