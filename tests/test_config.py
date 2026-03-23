from pathlib import Path

import pytest

from src.utils.config import load_config


def test_load_config_returns_dict():
    config = load_config()
    assert isinstance(config, dict)


def test_load_config_has_required_keys():
    config = load_config()
    for key in ["project", "data", "aws", "training", "mlflow", "serving"]:
        assert key in config


def test_load_config_custom_path(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("project:\n  name: test\n")
    config = load_config(config_file)
    assert config["project"]["name"] == "test"


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/ruta/que/no/existe/config.yaml")
