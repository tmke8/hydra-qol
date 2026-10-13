"""Checking that all YAML config files in a directory are valid.

YAML config files are not checked by the type checker, so if, for example, a field in
one of the config dataclasses is renamed, a YAML file that still uses the old name would
only fail once someone tries to use it. The functions in this module catch that without
running any experiments: they compose every config file with Hydra (which validates keys
and value types against the registered dataclasses) and then instantiate the main config
object from the result.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
import traceback
from typing import Any, TextIO

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from .registration import register_hydra_config

__all__ = [
    "CheckResult",
    "check_configs",
    "construct_override",
    "to_object",
    "validate_configs",
]

_MISSING = object()


@dataclass(frozen=True, kw_only=True)
class CheckResult:
    """Outcome of checking a single YAML config file."""

    path: Path
    """Path of the config file, relative to the config directory."""
    cmdline_arg: str
    """The command-line argument that would select this config file.

    This is ``--config-name <name>`` for primary configs and ``<group>=<option>`` or
    ``+<group>=<option>`` for config-group options.
    """
    error: Exception | None = None
    """The exception that was raised while composing or instantiating the config, if
    any."""
    traceback: str | None = None
    """The formatted traceback of :attr:`error`, if any."""

    @property
    def ok(self) -> bool:
        """Whether the config file is valid."""
        return self.error is None

    @property
    def description(self) -> str:
        """Short, human-readable description of the config file."""
        return f"{self.path.as_posix()} ({self.cmdline_arg})"


def validate_configs(
    main_cls: type,
    groups: dict[str, dict[str, type]],
    *,
    config_name: str,
    config_dir: Path | str,
) -> list[CheckResult]:
    """Compose and instantiate every YAML config file in ``config_dir``.

    First, ``main_cls`` and ``groups`` are registered with
    :func:`~hydra_qol.register_hydra_config` (with ``replace=True``, so that it doesn't
    matter whether the main script already registered them). Then, every ``.yaml``/``.yml`` file below
    ``config_dir`` is composed with Hydra: files at the top level of ``config_dir`` are
    treated as primary configs (``--config-name <name>``), all other files are treated
    as options of a config group (``<group>=<option>``). Finally, the composed config is
    converted to an instance of ``main_cls`` with :func:`to_object`. Any exception
    raised along the way is recorded in the corresponding :class:`CheckResult`.

    :param main_cls: The main config class; has to be a dataclass.
    :param groups: The config groups, in the same format as expected by
        :func:`~hydra_qol.register_hydra_config`.
    :param config_name: Name of the main schema. This has to match both the
        ``schema_name`` passed to :func:`~hydra_qol.register_hydra_config` and the
        ``config_name`` passed to ``@hydra.main()`` in the main script.
    :param config_dir: The directory containing the YAML config files; the same
        directory as the ``config_path`` passed to ``@hydra.main()``.
    :returns: One :class:`CheckResult` per config file, sorted by path.
    :raises ValueError: If the config classes themselves are malformed (see
        :func:`~hydra_qol.register_hydra_config`).
    """
    config_dir = Path(config_dir).resolve()
    register_hydra_config(main_cls, groups, schema_name=config_name, replace=True)

    results: list[CheckResult] = []
    with initialize_config_dir(config_dir=str(config_dir)):
        # Groups that are already in the defaults list are overridden with
        # `group=option`, all others have to be appended with `+group=option`.
        base_cfg = compose(config_name, return_hydra_config=True)
        groups_in_defaults = set(base_cfg.hydra.runtime.choices)

        for path in sorted(config_dir.rglob("*.y*ml")):
            # For each YAML file, we construct an override like `group=sub/config`.
            rel_path = path.relative_to(config_dir)
            name_parts = rel_path.with_suffix("").parts

            if len(name_parts) == 1:
                # A config at the top level is a primary config.
                primary_config = name_parts[0]
                overrides: list[str] = []
                cmdline_arg = f"--config-name {primary_config}"
            else:
                # Any other config is an option in a config group.
                override = construct_override(name_parts, groups_in_defaults)
                primary_config = config_name
                overrides = [override]
                cmdline_arg = override

            try:
                cfg = compose(
                    primary_config, overrides=overrides, return_hydra_config=True
                )
                config_obj = to_object(cfg)
                if not isinstance(config_obj, main_cls):
                    raise TypeError(
                        f"Config was not instantiated as `{main_cls.__name__}`, but as "
                        f"`{type(config_obj).__name__}`. (Does the primary config "
                        f"inherit from `{config_name}` in its defaults list?)"
                    )
            except Exception as exc:
                results.append(
                    CheckResult(
                        path=rel_path,
                        cmdline_arg=cmdline_arg,
                        error=exc,
                        traceback=traceback.format_exc(),
                    )
                )
            else:
                results.append(CheckResult(path=rel_path, cmdline_arg=cmdline_arg))
    return results


def check_configs(
    main_cls: type,
    groups: dict[str, dict[str, type]],
    *,
    config_name: str,
    config_dir: Path | str,
    file: TextIO | None = None,
) -> int:
    """Check all YAML config files in ``config_dir`` and print a report.

    This is a thin wrapper around :func:`validate_configs` that prints one line per
    config file (plus the traceback for every failure) and returns an exit code, so
    that it can be used directly in a script::

        if __name__ == "__main__":
            sys.exit(
                check_configs(
                    Config, CONFIG_GROUPS, config_name="main_config", config_dir="configs"
                )
            )

    :param main_cls: The main config class; has to be a dataclass.
    :param groups: The config groups, in the same format as expected by
        :func:`~hydra_qol.register_hydra_config`.
    :param config_name: Name of the main schema; see :func:`validate_configs`.
    :param config_dir: The directory containing the YAML config files.
    :param file: Where to print the report to; defaults to ``sys.stdout``.
    :returns: ``0`` if all config files are valid, ``1`` otherwise.
    """
    out = sys.stdout if file is None else file
    results = validate_configs(
        main_cls, groups, config_name=config_name, config_dir=config_dir
    )

    failures = [result for result in results if not result.ok]
    for result in results:
        if result.ok:
            print(f"OK    {result.description}", file=out)
        else:
            print(f"FAIL  {result.description}", file=out)
            print(result.traceback, file=out)

    if failures:
        print(f"\n{len(failures)} config file(s) are invalid:", file=out)
        for failure in failures:
            print(f"  - {failure.description}", file=out)
        return 1
    print("\nAll config files are valid.", file=out)
    return 0


def construct_override(name_parts: Sequence[str], groups_in_defaults: set[str]) -> str:
    """Construct the command-line override that selects a config-group option.

    Groups can be nested (e.g. ``hydra/launcher``) and options can contain slashes
    (e.g. ``experiment=cmnist/cnn``), so we pick the longest known group that is a
    prefix of the path and fall back to the top-level directory.

    :param name_parts: The parts of the path of the config file, relative to the config
        directory and without the file extension; e.g. ``("hydra", "launcher", "slurm")``
        for ``configs/hydra/launcher/slurm.yaml``.
    :param groups_in_defaults: The groups that are already in the defaults list. Options
        of these groups can be selected with ``group=option``; all other groups have to
        be added with ``+group=option``.
    :returns: The override string, e.g. ``"hydra/launcher=slurm"`` or ``"+opt=adamw"``.
    """
    split = next(
        (
            i
            for i in range(len(name_parts) - 1, 0, -1)
            if "/".join(name_parts[:i]) in groups_in_defaults
        ),
        1,
    )
    group = "/".join(name_parts[:split])
    option = "/".join(name_parts[split:])
    prefix = "" if group in groups_in_defaults else "+"
    return f"{prefix}{group}={option}"


def to_object(cfg: DictConfig) -> Any:
    """Instantiate the config object from a composed config.

    This mimics what ``@hydra.main()`` does before calling the decorated function
    (removing the ``hydra`` node) and then calls ``OmegaConf.to_object()``. In addition,
    the keys listed under ``hydra.sweeper.params`` are checked for existence, because
    they are just strings in the sweeper config and would otherwise only fail once a
    sweep is started.

    :param cfg: A config as returned by ``compose(..., return_hydra_config=True)``.
    :returns: The instantiated config object (an instance of the registered schema
        class, if the config was composed from it).
    :raises KeyError: If a key in ``hydra.sweeper.params`` does not exist in the config.
    """
    # Keys that are swept over by the sweeper are just strings in the sweeper config,
    # so we check here that they actually exist in the config.
    if (params := OmegaConf.select(cfg, "hydra.sweeper.params")) is not None:
        for key in params:
            if OmegaConf.select(cfg, key, default=_MISSING) is _MISSING:
                raise KeyError(f"Swept-over key '{key}' does not exist in the config.")

    # Remove the hydra node (as `@hydra.main` does) and instantiate the dataclasses.
    with open_dict(cfg):
        del cfg["hydra"]
    return OmegaConf.to_object(cfg)
