import torch

from cardiogb.losses import LossWeights
from cardiogb.models.neural_ode import NeuralODEFunc
from cardiogb.training import CrossSectionalTrainer, CrossSectionalTransition, TrainerConfig


def test_cross_sectional_trainer_accepts_unmatched_populations(tmp_path) -> None:
    model = NeuralODEFunc(hidden_dim=8, layers=1)
    transition = CrossSectionalTransition(
        source_states=torch.rand(12, 6),
        target_states=torch.rand(17, 6),
        graph=None,
        t0=0.0,
        t1=0.1,
    )
    trainer = CrossSectionalTrainer(
        model,
        device="cpu",
        config=TrainerConfig(epochs=2, step_size=0.1, early_stopping_patience=2),
        loss_weights=LossWeights(distribution=1.0),
    )
    history = trainer.fit([transition], [transition], checkpoint_path=tmp_path / "model.pt")
    assert len(history) == 2
    assert (tmp_path / "model.pt").is_file()
