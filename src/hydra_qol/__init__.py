"""Quality-of-life helpers for `Hydra <https://hydra.cc>`_.

The package has two parts:

* :func:`register_hydra_config` and :func:`prepare_for_logging` help with setting up
  a dataclass-based configuration (registering the schema and the config groups in
  Hydra's ``ConfigStore``, and turning a composed config into a plain dictionary for
  logging).
* :func:`check_configs` and :func:`validate_configs` compose every YAML file in a config
  directory and instantiate the config object from it, so that broken config files are
  caught without running any experiments.
"""

from .checking import CheckResult as CheckResult
from .checking import check_configs as check_configs
from .checking import construct_override as construct_override
from .checking import to_object as to_object
from .checking import validate_configs as validate_configs
from .formatting import prepare_for_logging as prepare_for_logging
from .registration import register_hydra_config as register_hydra_config

__all__ = [
    "CheckResult",
    "check_configs",
    "construct_override",
    "prepare_for_logging",
    "register_hydra_config",
    "to_object",
    "validate_configs",
]
