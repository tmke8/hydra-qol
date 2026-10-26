"""Quality-of-life helpers for `Hydra <https://hydra.cc>`_.

The package has two parts:

* :func:`register_hydra_config`, :func:`prepare_for_logging` and
  :func:`reconstruct_cmd` help with setting up a dataclass-based configuration
  (registering the schema and the config groups in Hydra's ``ConfigStore``, turning a
  composed config into a plain dictionary for logging, and reconstructing the command
  that started the program).
* :func:`check_configs` and :func:`validate_configs` compose every YAML file in a config
  directory and instantiate the config object from it, so that broken config files are
  caught without running any experiments.
"""

from .checking import *
from .logging import *
from .registration import *
