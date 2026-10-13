import dataclasses
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

from hydra import compose, initialize
from omegaconf import MISSING, MissingMandatoryValue, OmegaConf
import pytest

from hydra_qol import prepare_for_logging, register_hydra_config

from .example_config import CONFIG_GROUPS, Config


def test_logging_dict() -> None:
    class TrainingType(Enum):
        iter = auto()
        epoch = auto()

    @dataclass
    class DataModule:
        root: Path = MISSING

    @dataclass
    class Model:
        layers: int = 1

    @dataclass
    class Config:
        dm: DataModule = dataclasses.field(default_factory=DataModule)
        model: Model = dataclasses.field(default_factory=Model)
        train: TrainingType = TrainingType.iter

    hydra_config = OmegaConf.structured(Config)
    hydra_config.model.layers = 3

    with pytest.raises(MissingMandatoryValue):  # `dm.root` is missing
        prepare_for_logging(hydra_config)

    hydra_config.dm.root = Path()
    logging_dict = prepare_for_logging(hydra_config)

    assert logging_dict == {
        "dm/DataModule": {"root": Path()},
        "model/Model": {"layers": 3},
        "train": "iter",
    }

    logging_dict = prepare_for_logging(hydra_config, enum_to_str=False)
    assert logging_dict["train"] is TrainingType.iter


def test_resolves_interpolations() -> None:
    @dataclass
    class Config:
        seed: int = 1
        name: str = "run${seed}"

    hydra_config = OmegaConf.structured(Config)
    assert prepare_for_logging(hydra_config) == {"seed": 1, "name": "run1"}


def test_composed_config_shows_chosen_option() -> None:
    """The type suffix tells us which config-group option was chosen."""
    register_hydra_config(
        Config, CONFIG_GROUPS, schema_name="main_config", replace=True
    )

    with initialize():
        cfg = compose("main_config", overrides=["model=cnn", "model.activation=GELU"])

    logging_dict = prepare_for_logging(cfg)
    # Hydra removes the `defaults` list from the composed config.
    assert set(logging_dict) == {
        "dm/ColoredMNISTDataModule",
        "model/SimpleCNNFactory",
        "opt/OptimisationCfg",
        "seed",
        "gpu",
        "run_name",
    }
    assert logging_dict["model/SimpleCNNFactory"] == {
        "kernel_size": 5,
        "activation": "GELU",
    }
    assert logging_dict["run_name"] == "seed42"
