import torch

from cardiogb.models.mechanistic import MechanisticODE
from cardiogb.synthetic import simulate_system
from cardiogb.synthetic.recovery import recover_mechanistic_parameters
from cardiogb.utils.config import load_yaml


def test_synthetic_simulator_returns_truth_and_hidden_values() -> None:
    model = MechanisticODE.from_config(load_yaml("configs/mechanistic_model.yaml"))
    result = simulate_system(
        model,
        observation_times=[0.0, 0.25, 0.5],
        num_entities=16,
        hidden_mechanism=True,
        noise_std=0.01,
        step_size=0.05,
    )
    assert result.latent_clean.shape == (3, 16, 6)
    assert result.observations.shape == result.latent_clean.shape
    assert torch.count_nonzero(result.hidden_mechanism_values[..., 3]) > 0
    assert "alpha_A" in result.true_parameters


def test_parameter_recovery_can_run_multiple_backward_steps() -> None:
    truth = MechanisticODE.from_config(load_yaml("configs/mechanistic_model.yaml"))
    simulated = simulate_system(
        truth, observation_times=[0.0, 0.1, 0.2], num_entities=8, noise_std=0.0, step_size=0.1
    )
    fitted = MechanisticODE.from_config(load_yaml("configs/mechanistic_model.yaml"))
    recovered = recover_mechanistic_parameters(
        fitted,
        simulated.times,
        simulated.observations,
        simulated.true_parameters,
        epochs=2,
        step_size=0.1,
    )
    assert len(recovered.loss_history) == 2
