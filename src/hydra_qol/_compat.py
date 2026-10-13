"""Small shims that keep the package working the same across Hydra versions."""

import inspect
import re
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore

__all__ = ["VERSION_BASE_KWARGS", "store"]


def _hydra_version() -> tuple[int, int]:
    match = re.match(r"(\d+)\.(\d+)", hydra.__version__)
    return (int(match[1]), int(match[2])) if match else (0, 0)


# Hydra <1.4 warns if `version_base` is *not* passed to `initialize*()`/`@hydra.main()`,
# Hydra >=1.4 warns if it *is* passed. Either way, we want the current defaults.
VERSION_BASE_KWARGS: dict[str, Any] = (
    {} if _hydra_version() >= (1, 4) else {"version_base": None}
)

_STORE_HAS_REPLACE = "replace" in inspect.signature(ConfigStore.store).parameters


def store(
    cs: ConfigStore,
    *,
    name: str,
    node: Any,
    group: str | None = None,
    replace: bool | None,
) -> None:
    """Call ``cs.store()`` with Hydra 1.4's ``replace`` semantics on every Hydra version.

    On a collision (the name is already registered), ``replace=True`` replaces silently,
    ``replace=False`` raises :class:`ValueError`, and ``replace=None`` does whatever the
    installed Hydra does by default: 1.3 replaces silently, 1.4 warns and replaces, and
    1.5 is going to raise. Hydra 1.3 has no ``replace`` parameter, so ``False`` is
    emulated there.
    """
    if _STORE_HAS_REPLACE:
        # Passed as **kwargs because the parameter doesn't exist in Hydra 1.3's stubs.
        extra: dict[str, Any] = {"replace": replace}
        cs.store(name=name, node=node, group=group, **extra)
        return
    if replace is False:
        path = name if group is None else f"{group}/{name}"
        cur: dict[str, Any] = cs.repo
        for part in [] if group is None else group.split("/"):
            cur = cur.get(part, {})
        if f"{name}.yaml" in cur:
            raise ValueError(
                f"ConfigStore config '{path}.yaml' is already registered. "
                "Use replace=True to replace it."
            )
    cs.store(name=name, node=node, group=group)
