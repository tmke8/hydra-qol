"""Registering dataclass-based configs with Hydra's ``ConfigStore``."""

import dataclasses
from dataclasses import MISSING, is_dataclass
from typing import Any, Final, get_args, get_type_hints

from hydra.core.config_store import ConfigStore

from . import _compat

__all__ = ["register_hydra_config"]

_NEED: Final = "there should be"
_IF: Final = "if an entry has"


def register_hydra_config(
    main_cls: type,
    groups: dict[str, dict[str, type]],
    schema_name: str = "config_schema",
    *,
    replace: bool | None = None,
) -> None:
    """Check the given config classes and store everything in Hydra's ``ConfigStore``.

    This function performs two tasks: 1) make the necessary calls to ``ConfigStore``
    and 2) run some checks over the given config classes and, if there are problems,
    try to give a nice error message.

    The checks are based on the following rules:

    * An entry of type ``Any`` has to have registered variants (an entry in ``groups``)
      and may not have a default value.
    * An entry whose type is a dataclass and which has no default value has to have
      registered variants, and all variants have to be subclasses of the entry's type.
    * An entry whose type is a dataclass and which has a default value may not have
      registered variants.
    * An entry with registered variants has to have a dataclass type or ``Any``
      (unions of dataclasses are not supported by Hydra).
    * Variants may not have generic base classes.

    :param main_cls: The main config class; has to be a dataclass.
    :param groups: A dictionary that defines all the variants. The keys of the top level
        of the dictionary should correspond to the group names (which are also the
        names of the entries in ``main_cls``), and the keys in the nested dictionaries
        should correspond to the names of the options.
    :param schema_name: Name under which the main schema is stored. This name has to
        be passed as ``config_name`` to ``@hydra.main()`` (or it has to appear in the
        defaults list of a YAML file that is used as the primary config).
    :param replace: What to do if a config with the same name is already registered in
        the ``ConfigStore`` (this happens, for example, if this function is called
        twice). ``True`` replaces the existing config silently, ``False`` raises a
        :class:`ValueError`, and ``None`` (the default) leaves the decision to Hydra:
        Hydra 1.3 replaces silently, Hydra 1.4 warns and replaces, and Hydra 1.5 will
        raise. For a normal Hydra app, the default is the right choice, because a
        collision usually means that something is registered twice by accident.
    :raises ValueError: If the config is malformed in some way, or on a collision
        with ``replace=False``.
    :raises RuntimeError: If Hydra itself is throwing an error.

    :example:
        .. code-block:: python

            @dataclass
            class DataModule:
                root: Path = Path()

            @dataclass
            class LinearModel:
                dim: int = 256

            @dataclass
            class CNNModel:
                kernel: int = 3

            @dataclass
            class Config:
                dm: DataModule = field(default_factory=DataModule)
                model: Any

            groups = {"model": {"linear": LinearModel, "cnn": CNNModel}}
            register_hydra_config(Config, groups, schema_name="main_config")
    """
    assert isinstance(main_cls, type), "`main_cls` has to be a type."
    if not is_dataclass(main_cls):
        raise ValueError(f"The config class {main_cls.__name__} should be a dataclass.")
    entries = dataclasses.fields(main_cls)
    try:
        types = get_type_hints(main_cls)
    except NameError as exc:
        raise ValueError(
            f"Can't resolve type hints from the config class: `{main_cls.__name__}`."
        ) from exc

    for entry in entries:
        has_default = (
            entry.default is not MISSING or entry.default_factory is not MISSING
        )
        _check_entry(
            entry.name, types[entry.name], has_default=has_default, groups=groups
        )

    cs = ConfigStore.instance()
    _compat.store(cs, node=main_cls, name=schema_name, replace=replace)
    for group, variants in groups.items():
        for var_name, var_type in variants.items():
            bases = getattr(var_type, "__orig_bases__", None)
            if bases is not None and len(bases) > 0 and len(get_args(bases[0])) > 0:
                raise ValueError(
                    f"Can't register a dataclass with generic base class: `{var_type.__name__}`"
                    f" with base class `{bases[0].__name__}`."
                )
            try:
                _compat.store(
                    cs, node=var_type, name=var_name, group=group, replace=replace
                )
            except ValueError:
                # A collision with `replace=False`; the message is already descriptive.
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"{main_cls=}, {var_type=}, {var_name=}, {group=}"
                ) from exc


def _check_entry(
    name: str, typ: Any, *, has_default: bool, groups: dict[str, dict[str, type]]
) -> None:
    """Check a single field of the main config class against the registered variants."""
    group = groups.get(name)
    if typ == Any:
        if group is None:
            raise ValueError(f"{_IF} type `Any`, {_NEED} variants: `{name}`")
        for var_name, var_class in group.items():
            if not is_dataclass(var_class):
                raise ValueError(
                    f"All variants should be dataclasses: type `{var_class.__name__}` "
                    f"of variant `{name}={var_name}` is not a dataclass."
                )
        if has_default:
            raise ValueError(f"{_IF} type `Any`, {_NEED} no default value: `{name}`")
    elif is_dataclass(typ):
        if has_default:
            if group is not None:
                raise ValueError(
                    f"Can't have both a default value and variants: `{name}`."
                )
        elif group is not None:
            for var_name, var_class in group.items():
                if not issubclass(var_class, typ):
                    raise ValueError(
                        f"All variants should be subclasses of their entry's type: type"
                        f" `{var_class.__name__}` of variant `{name}={var_name}` "
                        f"is not a subclass of `{typ.__name__}`."
                    )
        else:
            raise ValueError(
                f"{_IF} a dataclass type, "
                f"{_NEED} a default value or registered variants: `{name}`. "
                "You can specify a default value with `field(default_factory=...)`."
            )
    elif group is not None:
        raise ValueError(
            f"Entry `{name}` has registered variants, but its type annotation, "
            f"`{getattr(typ, '__name__', str(typ))}`, is not a dataclass. "
            "(Note that unions of dataclasses are not allowed either.) "
            "You can always use `Any` for the type annotation."
        )
