from cardiogb.models.factory import build_model
from cardiogb.utils.config import load_yaml


def test_factory_builds_all_five_models() -> None:
    model_config = load_yaml("configs/model.yaml")
    mechanistic = load_yaml("configs/mechanistic_model.yaml")
    for name in ["persistence", "mechanistic_ode", "neural_ode", "graph_neural_ode", "cardiogb"]:
        assert build_model(name, model_config, mechanistic) is not None
