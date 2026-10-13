"""Small shims that keep the package warning-free across Hydra versions."""

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


def store(cs: ConfigStore, **kwargs: Any) -> None:
    """Call ``cs.store(**kwargs)``, replacing an already registered config.

    Hydra 1.3 silently replaces; Hydra 1.4 warns unless ``replace=True`` is passed, and
    Hydra 1.5 will raise. Replacing is what we want: registering the same config twice
    (e.g. in a test suite, or when checking the configs of an already imported main
    module) should be idempotent.
    """
    if _STORE_HAS_REPLACE:
        kwargs["replace"] = True
    cs.store(**kwargs)
