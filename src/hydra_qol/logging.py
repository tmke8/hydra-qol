"""Turning a composed Hydra config into something that is nice to log."""

import shlex
from typing import Any, cast

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

__all__ = ["prepare_for_logging", "reconstruct_cmd"]


def prepare_for_logging(
    hydra_config: DictConfig, *, enum_to_str: bool = True
) -> dict[str, Any]:
    """Take a Hydra config and make it prettier for logging.

    Things this function does: turn enums into strings, resolve any interpolations, and
    mark subconfigs with their type. The last point is the main reason to use this
    function: if an entry of the config is a config group with several options (say, a
    ``model`` entry that can be ``fcn`` or ``cnn``), the key of that entry in the
    returned dictionary becomes ``"model/FcnFactory"`` or ``"model/SimpleCNNFactory"``,
    so that the chosen option is visible in the logs.

    :param hydra_config: The config object as received by the function decorated with
        ``@hydra.main()`` (or as returned by :func:`hydra.compose`).
    :param enum_to_str: Whether to convert enum values to strings (their ``name``).
    :returns: A plain (nested) dictionary with resolved values.
    :raises omegaconf.MissingMandatoryValue: If the config contains an entry that is
        still ``MISSING`` (``???``).

    :example:
        .. code-block:: python

            @hydra.main(config_path="configs", config_name="main_config")
            def main(hydra_config: DictConfig) -> None:
                config_for_logging = prepare_for_logging(hydra_config)
                wandb.init(config=config_for_logging)
    """
    raw_config = OmegaConf.to_container(
        hydra_config, throw_on_missing=True, enum_to_str=enum_to_str, resolve=True
    )
    assert isinstance(raw_config, dict)
    raw_config = cast(dict[str, Any], raw_config)
    return {
        f"{key}/{_type_name(dict_)}"
        if isinstance(dict_ := hydra_config[key], DictConfig)
        else key: value
        for key, value in raw_config.items()
    }


def _type_name(cfg: DictConfig) -> str:
    """Name of the dataclass backing a config node (or of the node itself if untyped)."""
    typ = OmegaConf.get_type(cfg)
    return str(typ.__name__) if typ is not None else type(cfg).__name__


def reconstruct_cmd() -> str:
    """Reconstruct the python command that was used to start this program.

    The command consists of the name of the script followed by the overrides that were
    given on the command line, quoted such that the command can be pasted into a shell.
    This is useful for logging how a run can be reproduced. For
    ``python main.py model=cnn seed=1``, the returned string is
    ``"main.py model=cnn seed=1"``.

    The script name is derived from Hydra's job name (``hydra.job.name``), which by
    default is the name of the script without the ``.py`` suffix. Overrides of Hydra's
    own config (``hydra.*``) and flags like ``--multirun`` are not included.

    :returns: The script name and the overrides, joined with :func:`shlex.join`.
    :raises ValueError: If called outside of a function decorated with
        ``@hydra.main()``.

    :example:
        .. code-block:: python

            @hydra.main(config_path="configs", config_name="main_config")
            def main(hydra_config: DictConfig) -> None:
                wandb.init(notes=reconstruct_cmd())
    """
    internal_config = HydraConfig.get()
    program = internal_config.job.name + ".py"
    args = OmegaConf.to_container(internal_config.overrides.task)
    assert isinstance(args, list)
    return shlex.join([program] + args)
