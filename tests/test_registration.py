import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar
import warnings

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import pytest

from hydra_qol import register_hydra_config
from hydra_qol._compat import VERSION_BASE_KWARGS

from .example_config import (
    CONFIG_GROUPS,
    ColoredMNISTDataModule,
    Config,
    FcnFactory,
    SimpleCNNFactory,
)


def test_not_a_dataclass() -> None:
    class Config:
        seed: int = 1

    with pytest.raises(ValueError, match="should be a dataclass"):
        register_hydra_config(Config, {}, replace=True)


def test_config_no_default() -> None:
    """This isn't so much wrong as just clumsy."""

    @dataclass
    class DataModule:
        root: Path

    @dataclass
    class Config:
        dm: DataModule

    options: dict[str, dict[str, type]] = {}
    with pytest.raises(ValueError, match="default value or registered variants"):
        register_hydra_config(Config, options, replace=True)

    options = {"dm": {"base": DataModule}}
    register_hydra_config(Config, options, replace=True)

    @dataclass
    class UnrelatedClass:
        root: Path

    options = {"dm": {"base": UnrelatedClass}}
    with pytest.raises(ValueError, match="is not a subclass of `DataModule`"):
        register_hydra_config(Config, options, replace=True)


def test_config_any() -> None:
    @dataclass
    class DataModule:
        root: Path

    @dataclass
    class Config:
        dm: Any

    # we're assuming that the only reason you want to use Any is that
    # you want to use variants
    options: dict[str, dict[str, type]] = {}
    with pytest.raises(ValueError, match="there should be variants"):
        register_hydra_config(Config, options, replace=True)

    options = {"dm": {"base": DataModule}}
    register_hydra_config(Config, options, replace=True)

    class NotDC:
        x: int

    options = {"dm": {"base": NotDC}}
    with pytest.raises(ValueError, match="All variants should be dataclasses"):
        register_hydra_config(Config, options, replace=True)


def test_config_any_string() -> None:
    @dataclass
    class DataModule:
        root: Path

    @dataclass
    class Config:
        dm: "Any"

    options: dict[str, dict[str, type]] = {}
    with pytest.raises(ValueError, match="there should be variants"):
        register_hydra_config(Config, options, replace=True)

    options = {"dm": {"base": DataModule}}
    register_hydra_config(Config, options, replace=True)


def test_config_base_class() -> None:
    @dataclass
    class DataModule:
        root: Path

    @dataclass
    class CMnist(DataModule):
        colorize: bool

    @dataclass
    class CelebA(DataModule):
        target: str

    @dataclass
    class Config:
        dm: DataModule

    options: dict[str, dict[str, type]] = {}
    with pytest.raises(ValueError):
        register_hydra_config(Config, options, replace=True)

    options = {"dm": {"cmnist": CMnist, "celeba": CelebA}}
    register_hydra_config(Config, options, replace=True)

    @dataclass
    class NotSubclass:
        root: Path

    options = {"dm": {"base": NotSubclass}}
    with pytest.raises(ValueError, match="is not a subclass of"):
        register_hydra_config(Config, options, replace=True)


def test_config_union() -> None:
    @dataclass
    class CMnist:
        colorize: bool

    @dataclass
    class CelebA:
        target: str

    @dataclass
    class Config:
        # This is unfortunately not supported.
        dm: CMnist | CelebA

    options: dict[str, dict[str, type]] = {"dm": {"cmnist": CMnist, "celeba": CelebA}}
    with pytest.raises(ValueError, match="is not a dataclass"):
        register_hydra_config(Config, options, replace=True)


def test_config_any_with_default() -> None:
    """An Any field with default is completely out."""

    @dataclass
    class Model:
        layers: int = 1

    @dataclass
    class Config:
        model: Any = dataclasses.field(default_factory=Model)

    options: dict[str, dict[str, type]] = {}
    with pytest.raises(ValueError):
        register_hydra_config(Config, options, replace=True)

    options = {"model": {"base": Model}}
    with pytest.raises(ValueError, match="there should be no default value"):
        register_hydra_config(Config, options, replace=True)


def test_config_with_default() -> None:
    """A normal field with a default should not have variants."""

    @dataclass
    class Model:
        layers: int = 1

    @dataclass
    class Config:
        model: Model = dataclasses.field(default_factory=Model)

    options: dict[str, dict[str, type]] = {}
    register_hydra_config(Config, options, replace=True)

    options = {"model": {"base": Model}}
    with pytest.raises(ValueError, match="both a default value and variants"):
        register_hydra_config(Config, options, replace=True)


@dataclass
class _GlobalModel:
    layers: int = 1


def test_config_with_default_string() -> None:
    """Need to use a global class here because otherwise the type annotations can't be resolved."""

    @dataclass
    class Config:
        model: "_GlobalModel" = dataclasses.field(default_factory=_GlobalModel)

    options: dict[str, dict[str, type]] = {}
    register_hydra_config(Config, options, replace=True)

    options = {"model": {"base": _GlobalModel}}
    with pytest.raises(ValueError):
        register_hydra_config(Config, options, replace=True)


T = TypeVar("T")


def test_config_generic() -> None:
    class Base(Generic[T]): ...

    @dataclass
    class DataModule(Base):
        root: Path

    @dataclass
    class Config:
        dm: DataModule

    options: dict[str, dict[str, type]] = {"dm": {"asdf": DataModule}}
    with pytest.raises(ValueError, match="generic base class"):
        register_hydra_config(Config, options, replace=True)


def test_stored_in_config_store() -> None:
    register_hydra_config(
        Config, CONFIG_GROUPS, schema_name="main_config", replace=True
    )

    repo = ConfigStore.instance().repo
    assert "main_config.yaml" in repo
    assert OmegaConf.get_type(repo["main_config.yaml"].node) is Config
    assert set(repo["dm"]) >= {"celeba.yaml", "cmnist.yaml"}
    assert set(repo["model"]) >= {"fcn.yaml", "cnn.yaml"}
    assert OmegaConf.get_type(repo["model"]["cnn.yaml"].node) is SimpleCNNFactory


def test_compose_from_registered_schema() -> None:
    """End-to-end: after registration, Hydra can compose the config without any YAML."""
    register_hydra_config(
        Config, CONFIG_GROUPS, schema_name="main_config", replace=True
    )

    with initialize(**VERSION_BASE_KWARGS):
        cfg = compose("main_config")
        config = OmegaConf.to_object(cfg)
        assert isinstance(config, Config)
        assert isinstance(config.dm, ColoredMNISTDataModule)
        assert isinstance(config.model, FcnFactory)
        assert config.seed == 42

        cfg = compose(
            "main_config", overrides=["model=cnn", "model.kernel_size=3", "seed=3"]
        )
        config = OmegaConf.to_object(cfg)
        assert isinstance(config, Config)
        assert isinstance(config.model, SimpleCNNFactory)
        assert config.model.kernel_size == 3
        assert config.run_name == "seed3"

        # Keys that don't exist in the selected option are rejected.
        with pytest.raises(Exception, match="hidden_dim"):
            compose("main_config", overrides=["model=cnn", "model.hidden_dim=10"])


def test_replace() -> None:
    @dataclass
    class Model:
        layers: int = 1

    @dataclass
    class Config:
        model: Model

    groups: dict[str, dict[str, type]] = {"model": {"replace_test": Model}}

    # First registration: nothing to replace, so `replace=False` is fine.
    register_hydra_config(Config, groups, schema_name="replace_test", replace=False)

    # Second registration: with `replace=False`, both the schema and the group option
    # collide.
    with pytest.raises(ValueError, match="'replace_test.yaml' is already registered"):
        register_hydra_config(Config, groups, schema_name="replace_test", replace=False)
    with pytest.raises(
        ValueError, match="'model/replace_test.yaml' is already registered"
    ):
        register_hydra_config(
            Config, groups, schema_name="replace_test_2", replace=False
        )

    # With `replace=True`, re-registering is silent on every Hydra version.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        register_hydra_config(Config, groups, schema_name="replace_test", replace=True)
