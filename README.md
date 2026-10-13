# hydra-qol

Quality-of-life helpers for [Hydra](https://hydra.cc): register dataclass-based configs
in one call, prepare composed configs for logging, and check that *all* YAML config
files in a project are valid without running any experiments.

## Installation

```bash
uv add hydra-qol
# or
pip install hydra-qol
```

Requires Python 3.11+, `hydra-core` 1.4+ (a pre-release at the time of writing) and
`omegaconf` 2.4+.

## What's in the box

| Function | Purpose |
| --- | --- |
| `register_hydra_config(Config, groups, schema_name=...)` | Stores the main config class and all config-group options in Hydra's `ConfigStore`, after checking the classes for common mistakes. |
| `prepare_for_logging(hydra_config)` | Turns a composed config into a plain dict (enums → strings, interpolations resolved, subconfigs tagged with their type), ready for `wandb.init(config=...)`. |
| `check_configs(Config, groups, config_name=..., config_dir=...)` | Composes every YAML file in `config_dir` and instantiates `Config` from it; prints a report and returns an exit code. |
| `validate_configs(...)` | Same as `check_configs`, but returns a list of `CheckResult` objects instead of printing. Handy in a test suite. |

## Quick start

```python
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
    ...


if __name__ == "__main__":
    register_hydra_config(Config, CONFIG_GROUPS, schema_name="main_config")
    main()
```

```python
# check_configs.py
import sys

from hydra_qol import check_configs

from my_project.config import CONFIG_GROUPS, Config

if __name__ == "__main__":
    sys.exit(
        check_configs(
            Config, CONFIG_GROUPS, config_name="main_config", config_dir="configs"
        )
    )
```

```
$ python check_configs.py
OK    dm/celeba_male_smiling.yaml (dm=celeba_male_smiling)
OK    good_run.yaml (--config-name good_run)
FAIL  model/single_linear_layer.yaml (model=single_linear_layer)
Traceback (most recent call last):
  ...
hydra.errors.ConfigCompositionException: Could not override 'model.num_hiden'.

1 config file(s) are invalid:
  - model/single_linear_layer.yaml (model=single_linear_layer)
```

A complete, annotated example project that uses these functions can be found at
[wearepal/hydra-example](https://github.com/wearepal/hydra-example).
The full documentation lives in [`docs/`](docs/) and is built with Sphinx:

```bash
uv sync --group docs
uv run sphinx-build -b html docs docs/_build/html
```

## Development

```bash
uv sync                # installs test, lint and docs dependencies
uv run pytest
uv run ruff check && uv run ruff format --check
uv run ty check
```

## Acknowledgements

`register_hydra_config` and `prepare_for_logging` originate from
[ranzen](https://github.com/wearepal/ranzen) (Apache 2.0), and the config-checking
functions from [hydra-example](https://github.com/wearepal/hydra-example).

## License

Apache 2.0, see [LICENSE](LICENSE).
