"""Config inheritance resolves sibling YAML files."""

from paths import project_root
from config import load_config


def test_quick_yaml_extends_base():
    cfg = load_config(project_root() / "configs" / "quick.yaml")
    assert cfg["experiment"]["profile"] == "quick"
    assert cfg["training"]["max_epochs"] == 200
    assert "weak_mlp" in cfg["experiment"]["methods"]
    assert "mlp_state" in cfg["experiment"]["methods"]
    assert cfg["training"]["lambda_boundary"] == 0.0
    assert cfg["vehicle"]["config"] == "configs/vehicle_twizy.yaml"
    assert cfg["windows"]["soc_event_min_duration_s"] == 60.0
