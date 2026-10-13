"""Config classes for a small example project, modelled after wearepal/hydra-example."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Final

__all__ = [
    "CONFIG_GROUPS",
    "Activation",
    "CelebADataModule",
    "ColoredMNISTDataModule",
    "Config",
    "DataModule",
    "FcnFactory",
    "ModelFactory",
    "OptimisationCfg",
    "SimpleCNNFactory",
]


class Activation(Enum):
    RELU = auto()
    GELU = auto()
    SELU = auto()


@dataclass(kw_only=True, eq=False)
class DataModule(ABC):
    root: Path = Path("/data")

    @property
    @abstractmethod
    def in_dim(self) -> int: ...


@dataclass(kw_only=True, eq=False)
class CelebADataModule(DataModule):
    default_res: int = 224
    superclass: str = "Blond_Hair"
    subclass: str = "Male"
    download: bool = False

    @property
    def in_dim(self) -> int:
        return self.default_res


@dataclass(kw_only=True, eq=False)
class ColoredMNISTDataModule(DataModule):
    default_res: int = 28
    num_colors: int = 10
    val_prop: float = 0.2

    @property
    def in_dim(self) -> int:
        return self.default_res


@dataclass(eq=False)
class ModelFactory(ABC):
    @abstractmethod
    def build(self, in_dim: int) -> str: ...


@dataclass(eq=False, kw_only=True)
class FcnFactory(ModelFactory):
    num_hidden: int = 1
    hidden_dim: int | None = None
    activation: Activation = Activation.GELU
    dropout_prob: float = 0.0

    def build(self, in_dim: int) -> str:
        return f"fcn({in_dim})"


@dataclass(eq=False, kw_only=True)
class SimpleCNNFactory(ModelFactory):
    kernel_size: int = 5
    activation: Activation = Activation.RELU

    def build(self, in_dim: int) -> str:
        return f"cnn({in_dim})"


@dataclass
class OptimisationCfg:
    lr: float = 5.0e-5
    weight_decay: float = 0.0


CONFIG_GROUPS: Final[dict[str, dict[str, type]]] = {
    "dm": {"celeba": CelebADataModule, "cmnist": ColoredMNISTDataModule},
    "model": {"fcn": FcnFactory, "cnn": SimpleCNNFactory},
}


@dataclass(kw_only=True)
class Config:
    # Hydra's defaults list. Hydra removes it from the composed config, but
    # `OmegaConf.to_object()` fills it in again from `default_factory`.
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"dm": "cmnist"},
            {"model": "fcn"},
            {"opt": None},
        ]
    )

    dm: DataModule
    model: ModelFactory
    opt: OptimisationCfg = field(default_factory=OptimisationCfg)
    seed: int = 42
    gpu: int = 0
    # An interpolation, to check that values get resolved.
    run_name: str = "seed${seed}"
