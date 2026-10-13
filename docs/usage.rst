Usage
=====

This page walks through a small project that uses ``hydra-qol``. It follows the layout
of `wearepal/hydra-example <https://github.com/wearepal/hydra-example>`_::

   my_project/
   ├── main.py
   ├── check_configs.py
   ├── my_project/
   │   └── config.py
   └── configs/
       ├── good_run.yaml
       ├── dm/
       │   └── celeba_male_smiling.yaml
       ├── model/
       │   └── single_linear_layer.yaml
       └── hydra/
           └── sweeper/
               └── fcn_params.yaml

Defining the config classes
---------------------------

The configuration is defined with dataclasses. Components that come in several
variants (here: the data module and the model) get an abstract base class and one
subclass per variant; these are the *config groups*. Components with a single variant
simply get a default value.

.. code-block:: python

   # my_project/config.py
   from abc import ABC
   from dataclasses import dataclass, field
   from typing import Any


   @dataclass(kw_only=True, eq=False)
   class DataModule(ABC):
       root: str = "/data"


   @dataclass(kw_only=True, eq=False)
   class CelebADataModule(DataModule):
       superclass: str = "Blond_Hair"
       subclass: str = "Male"


   @dataclass(kw_only=True, eq=False)
   class ColoredMNISTDataModule(DataModule):
       num_colors: int = 10


   @dataclass(eq=False)
   class ModelFactory(ABC): ...


   @dataclass(eq=False, kw_only=True)
   class FcnFactory(ModelFactory):
       num_hidden: int = 1
       hidden_dim: int | None = None


   @dataclass(eq=False, kw_only=True)
   class SimpleCNNFactory(ModelFactory):
       kernel_size: int = 5


   @dataclass
   class OptimisationCfg:
       lr: float = 5.0e-5
       weight_decay: float = 0.0


   # The keys of the outer dict are the group names, which have to match the field
   # names in `Config`. The keys of the inner dicts are the option names.
   CONFIG_GROUPS: dict[str, dict[str, type]] = {
       "dm": {"celeba": CelebADataModule, "cmnist": ColoredMNISTDataModule},
       "model": {"fcn": FcnFactory, "cnn": SimpleCNNFactory},
   }


   @dataclass(kw_only=True)
   class Config:
       # Hydra's defaults list. `_self_` has to come first, so that the types of the
       # selected config-group options are not overwritten by the abstract base types.
       defaults: list[Any] = field(
           default_factory=lambda: [
               "_self_",
               {"dm": "cmnist"},
               {"model": "fcn"},
               # Listing a non-group subconfig with `None` makes it possible to write
               # `opt=...` instead of `+opt=...` on the command line.
               {"opt": None},
           ]
       )

       # Config groups: no default value here; the default is in the defaults list.
       dm: DataModule
       model: ModelFactory

       # Normal subconfigs and fields with defaults.
       opt: OptimisationCfg = field(default_factory=OptimisationCfg)
       seed: int = 42
       gpu: int = 0

Registering the config
----------------------

:func:`~hydra_qol.register_hydra_config` stores ``Config`` under the given schema name
and every option of every group in Hydra's ``ConfigStore``. Before doing so, it checks
the config classes and raises a :class:`ValueError` with a descriptive message if, for
example, a config group has a default value *and* registered options, or if an option
is not a subclass of the entry's type. The schema name has to match the
``config_name`` that is passed to ``@hydra.main()``:

.. code-block:: python

   # main.py
   import hydra
   from omegaconf import DictConfig, OmegaConf
   from hydra_qol import prepare_for_logging, register_hydra_config

   from my_project.config import CONFIG_GROUPS, Config


   @hydra.main(config_path="configs", config_name="main_config")
   def main(hydra_config: DictConfig) -> None:
       config = OmegaConf.to_object(hydra_config)
       assert isinstance(config, Config)

       config_for_logging = prepare_for_logging(hydra_config)
       # e.g. wandb.init(config=config_for_logging)
       ...


   if __name__ == "__main__":
       register_hydra_config(Config, CONFIG_GROUPS, schema_name="main_config")
       main()

If a config name is already present in the ``ConfigStore`` (for example because
``register_hydra_config()`` is called twice), the ``replace`` argument decides what
happens: ``True`` replaces silently, ``False`` raises, and the default ``None`` leaves
it to Hydra, which warns (and will raise in Hydra 1.5), since an accidental double
registration is usually a bug. :func:`~hydra_qol.validate_configs` always registers
with ``replace=True``, so it can be run after the main module has been imported.

Now the config can be composed from the command line as usual:

.. code-block:: bash

   python main.py model=cnn model.kernel_size=3 seed=1

Preparing the config for logging
--------------------------------

:func:`~hydra_qol.prepare_for_logging` converts the composed config to a plain
dictionary: interpolations are resolved, enums become strings, and subconfigs are
marked with their type. The last point makes the chosen config-group options visible
in the logs. For ``python main.py model=cnn`` the dictionary looks like this:

.. code-block:: python

   {
       "dm/ColoredMNISTDataModule": {"root": "/data", "num_colors": 10},
       "model/SimpleCNNFactory": {"kernel_size": 5},
       "opt/OptimisationCfg": {"lr": 5e-05, "weight_decay": 0.0},
       "seed": 42,
       "gpu": 0,
   }

Config files
------------

YAML files in ``configs/`` extend the registered dataclasses. Files in a group
directory are options of that group and can inherit from registered options:

.. code-block:: yaml

   # configs/model/single_linear_layer.yaml
   defaults:
     - fcn

   num_hidden: 0

Files at the top level of ``configs/`` are *primary configs* that can be selected
with ``--config-name``. They should inherit from the schema so that Hydra validates
them:

.. code-block:: yaml

   # configs/good_run.yaml
   defaults:
     - main_config
     - override model: single_linear_layer
     - override dm: celeba_male_smiling
     - _self_

   seed: 1
   opt:
     lr: 0.001

Checking all config files
-------------------------

YAML files are not seen by the type checker. If a field is renamed, or a value of the
wrong type ends up in a YAML file, this is only noticed once someone selects that
file. :func:`~hydra_qol.check_configs` catches such problems without running any
experiment: it composes every YAML file below the config directory and instantiates
``Config`` from the result.

* Top-level files are composed as primary configs (``--config-name <name>``).
* Files in subdirectories are composed as config-group options on top of the schema
  (``<group>=<option>``, or ``+<group>=<option>`` if the group is not in the defaults
  list). Nested groups such as ``hydra/sweeper`` and options with slashes such as
  ``+experiment=cmnist/cnn`` are handled.
* The keys listed under ``hydra.sweeper.params`` are checked for existence, since they
  are plain strings in the sweeper config.
* The composed config has to be an instance of the main config class. A primary config
  that doesn't inherit from the schema fails this check, because Hydra wouldn't
  validate its keys.

.. code-block:: python

   # check_configs.py
   import sys

   from hydra_qol import check_configs

   from my_project.config import CONFIG_GROUPS, Config

   if __name__ == "__main__":
       sys.exit(
           check_configs(
               Config,
               CONFIG_GROUPS,
               # Has to match both `schema_name` and `config_name` in `main.py`.
               config_name="main_config",
               config_dir="configs",
           )
       )

The script prints one line per file and the traceback for every failure::

   OK    dm/celeba_male_smiling.yaml (dm=celeba_male_smiling)
   OK    good_run.yaml (--config-name good_run)
   OK    hydra/sweeper/fcn_params.yaml (hydra/sweeper=fcn_params)
   FAIL  model/single_linear_layer.yaml (model=single_linear_layer)
   Traceback (most recent call last):
     ...
   hydra.errors.ConfigCompositionException: Could not override 'model.num_hiden'.
   ...

   1 config file(s) are invalid:
     - model/single_linear_layer.yaml (model=single_linear_layer)

The exit code is ``1`` if any file is invalid, so the script can be run as a CI step:

.. code-block:: yaml

   - name: Check that all config files are valid
     run: uv run python check_configs.py

Alternatively, use :func:`~hydra_qol.validate_configs` from a test. It returns a list
of :class:`~hydra_qol.CheckResult` objects instead of printing:

.. code-block:: python

   # tests/test_configs.py
   from pathlib import Path

   import pytest
   from hydra_qol import validate_configs

   from my_project.config import CONFIG_GROUPS, Config

   CONFIG_DIR = Path(__file__).parents[1] / "configs"


   def test_all_configs_are_valid() -> None:
       results = validate_configs(
           Config, CONFIG_GROUPS, config_name="main_config", config_dir=CONFIG_DIR
       )
       failures = [r for r in results if not r.ok]
       assert not failures, "\n".join(f"{r.description}\n{r.traceback}" for r in failures)
