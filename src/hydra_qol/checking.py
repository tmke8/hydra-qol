"""Checking that all YAML config files in a directory are valid.

YAML config files are not checked by the type checker, so if, for example, a field in
one of the config dataclasses is renamed, a YAML file that still uses the old name would
only fail once someone tries to use it. The functions in this module catch that without
running any experiments: they compose every config file with Hydra (which validates keys
and value types against the registered dataclasses) and then instantiate the main config
object from the result.
"""

from collections.abc import Generator, Sequence, Set
from contextlib import contextmanager
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
    "ValidationContext",
    "check_configs",
    "construct_override",
    "setup_validation",
    "validate_config",
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


@dataclass(frozen=True, kw_only=True)
class ValidationContext:
    """State shared by all calls to :func:`validate_config`.

    Created by :func:`setup_validation`; only valid inside its ``with`` block.
    """

    main_cls: type
    """The main config class."""
    schema_name: str
    """Name under which :attr:`main_cls` is registered."""
    config_dir: Path
    """The resolved config directory."""
    config_with_defaults: str
    """Name of the primary config that the config-group options are composed on top
    of."""
    groups_in_defaults: frozenset[str]
    """The config groups that are in the defaults list of
    :attr:`config_with_defaults`."""


@contextmanager
def setup_validation(
    main_cls: type,
    groups: dict[str, dict[str, type]],
    *,
    schema_name: str,
    config_dir: Path | str,
    config_with_defaults: str | None = None,
) -> Generator[ValidationContext, None, None]:
    """Prepare for checking individual YAML config files with :func:`validate_config`.

    This registers ``main_cls`` and ``groups`` with
    :func:`~hydra_qol.register_hydra_config` (with ``replace=True``, so that it doesn't
    matter whether the main script already registered them) and initializes Hydra with
    ``config_dir``. Hydra is de-initialized again when the ``with`` block is left::

        with setup_validation(Config, CONFIG_GROUPS, ...) as context:
            for path in paths:
                result = validate_config(path, context)

    Use :func:`validate_configs` instead to simply check every file in ``config_dir``.

    :param main_cls: The main config class; has to be a dataclass.
    :param groups: The config groups, in the same format as expected by
        :func:`~hydra_qol.register_hydra_config`.
    :param schema_name: Name of the main schema; see :func:`validate_configs`.
    :param config_dir: The directory containing the YAML config files.
    :param config_with_defaults: Name of the primary config that the config-group
        options are composed on top of, if it differs from ``schema_name``; see
        :func:`validate_configs`.
    :returns: A context manager that yields the :class:`ValidationContext` to pass to
        :func:`validate_config`.
    :raises ValueError: If the config classes themselves are malformed (see
        :func:`~hydra_qol.register_hydra_config`).
    """
    config_dir = Path(config_dir).resolve()
    if config_with_defaults is None:
        config_with_defaults = schema_name
    register_hydra_config(main_cls, groups, schema_name=schema_name, replace=True)

    with initialize_config_dir(config_dir=str(config_dir)):
        # Groups that are already in the defaults list are overridden with
        # `group=option`, all others have to be appended with `+group=option`.
        base_cfg = compose(config_with_defaults, return_hydra_config=True)
        yield ValidationContext(
            main_cls=main_cls,
            schema_name=schema_name,
            config_dir=config_dir,
            config_with_defaults=config_with_defaults,
            groups_in_defaults=frozenset(base_cfg.hydra.runtime.choices),
        )


def validate_config(path: Path, context: ValidationContext) -> CheckResult:
    """Compose and instantiate a single YAML config file.

    Has to be called inside the ``with`` block of :func:`setup_validation`. See
    :func:`validate_configs` for how the file is composed and instantiated.

    :param path: Path of the config file; either relative to the config directory or
        an absolute path inside :attr:`ValidationContext.config_dir`.
    :param context: The context yielded by :func:`setup_validation`.
    :returns: The outcome of the check. Any exception raised while composing or
        instantiating the config is recorded in it.
    :raises ValueError: If ``path`` is absolute but not inside the config directory.
    """
    if path.is_absolute():
        path = path.relative_to(context.config_dir)
    # For each YAML file, we construct an override like `group=sub/config`.
    name_parts = path.with_suffix("").parts

    if len(name_parts) == 1:
        # A config at the top level is a primary config.
        primary_config = name_parts[0]
        overrides: list[str] = []
        cmdline_arg = f"--config-name {primary_config}"
    else:
        # Any other config is an option in a config group.
        override = construct_override(name_parts, context.groups_in_defaults)
        primary_config = context.config_with_defaults
        overrides = [override]
        cmdline_arg = override

    try:
        cfg = compose(primary_config, overrides=overrides, return_hydra_config=True)
        config_obj = _to_object(cfg)
        if not isinstance(config_obj, context.main_cls):
            raise TypeError(
                f"Config was not instantiated as `{context.main_cls.__name__}`, but as "
                f"`{type(config_obj).__name__}`. (Does the primary config inherit from "
                f"`{context.schema_name}` in its defaults list?)"
            )
    except Exception as exc:
        return CheckResult(
            path=path,
            cmdline_arg=cmdline_arg,
            error=exc,
            traceback=traceback.format_exc(),
        )
    return CheckResult(path=path, cmdline_arg=cmdline_arg)


def validate_configs(
    main_cls: type,
    groups: dict[str, dict[str, type]],
    *,
    schema_name: str,
    config_dir: Path | str,
    config_with_defaults: str | None = None,
) -> list[CheckResult]:
    """Compose and instantiate every YAML config file in ``config_dir``.

    First, ``main_cls`` and ``groups`` are registered with
    :func:`~hydra_qol.register_hydra_config` (with ``replace=True``, so that it doesn't
    matter whether the main script already registered them). Then, every
    ``.yaml``/``.yml`` file below ``config_dir`` is composed with Hydra: files at the
    top level of ``config_dir`` are treated as primary configs
    (``--config-name <name>``), all other files are treated as options of a config
    group (``<group>=<option>``) and are composed on top of ``config_with_defaults``.
    Finally, the composed config is converted to an instance of ``main_cls`` with
    :func:`to_object`. Any exception raised along the way is recorded in the
    corresponding :class:`CheckResult`.

    This is a shortcut for calling :func:`validate_config` on every file inside of
    :func:`setup_validation`.

    :param main_cls: The main config class; has to be a dataclass.
    :param groups: The config groups, in the same format as expected by
        :func:`~hydra_qol.register_hydra_config`.
    :param schema_name: Name under which ``main_cls`` is registered; the same as the
        ``schema_name`` passed to :func:`~hydra_qol.register_hydra_config` in the main
        script.
    :param config_dir: The directory containing the YAML config files; the same
        directory as the ``config_path`` passed to ``@hydra.main()``.
    :param config_with_defaults: Name of the primary config that the config-group
        options are composed on top of; usually the same as the ``config_name`` passed
        to ``@hydra.main()`` in the main script. This is either the schema itself (if
        the defaults list is part of ``main_cls``) or a YAML file in ``config_dir``
        whose defaults list starts with the schema. Every config-group option is
        composed on top of this config with only its own override (e.g.
        ``model=cnn``), so this config has to select a default option for *every*
        config group; otherwise, the groups that are not overridden stay missing and
        the check fails. Groups listed in its defaults list are overridden with
        ``group=option``, all others are added with ``+group=option``. Defaults to
        ``schema_name``.
    :returns: One :class:`CheckResult` per config file, sorted by path.
    :raises ValueError: If the config classes themselves are malformed (see
        :func:`~hydra_qol.register_hydra_config`).
    """
    with setup_validation(
        main_cls,
        groups,
        schema_name=schema_name,
        config_dir=config_dir,
        config_with_defaults=config_with_defaults,
    ) as context:
        return [
            validate_config(path, context)
            for path in sorted(context.config_dir.rglob("*.y*ml"))
        ]


def check_configs(
    main_cls: type,
    groups: dict[str, dict[str, type]],
    *,
    schema_name: str,
    config_dir: Path | str,
    config_with_defaults: str | None = None,
    file: TextIO | None = None,
) -> int:
    """Check all YAML config files in ``config_dir`` and print a report.

    This is a thin wrapper around :func:`validate_configs` that prints one line per
    config file (plus the traceback for every failure) and returns an exit code, so
    that it can be used directly in a script::

        if __name__ == "__main__":
            sys.exit(
                check_configs(
                    Config, CONFIG_GROUPS, schema_name="main_config", config_dir="configs"
                )
            )

    :param main_cls: The main config class; has to be a dataclass.
    :param groups: The config groups, in the same format as expected by
        :func:`~hydra_qol.register_hydra_config`.
    :param schema_name: Name of the main schema; see :func:`validate_configs`.
    :param config_dir: The directory containing the YAML config files.
    :param config_with_defaults: Name of the primary config that the config-group
        options are composed on top of, if it differs from ``schema_name``; see
        :func:`validate_configs`.
    :param file: Where to print the report to; defaults to ``sys.stdout``.
    :returns: ``0`` if all config files are valid, ``1`` otherwise.
    """
    out = sys.stdout if file is None else file
    results = validate_configs(
        main_cls,
        groups,
        schema_name=schema_name,
        config_dir=config_dir,
        config_with_defaults=config_with_defaults,
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


def construct_override(name_parts: Sequence[str], groups_in_defaults: Set[str]) -> str:
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


def _to_object(cfg: DictConfig) -> Any:
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
