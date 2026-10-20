from dataclasses import dataclass, field
import io
from pathlib import Path

from hydra.core.global_hydra import GlobalHydra
from hydra.errors import ConfigCompositionException, MissingConfigException
from omegaconf import OmegaConf
from omegaconf.errors import ValidationError
import pytest

from hydra_qol import (
    CheckResult,
    check_configs,
    construct_override,
    setup_validation,
    validate_config,
    validate_configs,
)
from hydra_qol.checking import _to_object

from .example_config import (
    CONFIG_GROUPS,
    Config,
    DataModule,
    ModelFactory,
    OptimisationCfg,
)

EXPECTED_CMDLINE_ARGS = {
    "dm/celeba_male_smiling.yaml": "dm=celeba_male_smiling",
    "dm/celeba_male_smiling_small.yaml": "dm=celeba_male_smiling_small",
    "experiment/cmnist/cnn.yaml": "+experiment=cmnist/cnn",
    "good_run.yaml": "--config-name good_run",
    "hydra/sweeper/grid.yaml": "hydra/sweeper=grid",
    "model/single_linear_layer.yaml": "model=single_linear_layer",
    "opt/weight_decay_adamw.yaml": "opt=weight_decay_adamw",
}


def _validate(config_dir: Path) -> dict[str, CheckResult]:
    results = validate_configs(
        Config, CONFIG_GROUPS, schema_name="main_config", config_dir=config_dir
    )
    return {result.path.as_posix(): result for result in results}


def test_all_valid(config_dir: Path) -> None:
    results = validate_configs(
        Config, CONFIG_GROUPS, schema_name="main_config", config_dir=config_dir
    )
    assert [r.path.as_posix() for r in results] == sorted(EXPECTED_CMDLINE_ARGS)
    assert {r.path.as_posix(): r.cmdline_arg for r in results} == EXPECTED_CMDLINE_ARGS
    for result in results:
        assert result.ok, result.traceback
        assert result.error is None
        assert result.traceback is None
        assert result.description == f"{result.path.as_posix()} ({result.cmdline_arg})"


def test_check_configs_all_valid(config_dir: Path) -> None:
    out = io.StringIO()
    exit_code = check_configs(
        Config,
        CONFIG_GROUPS,
        schema_name="main_config",
        config_dir=config_dir,
        file=out,
    )
    assert exit_code == 0
    lines = out.getvalue().splitlines()
    assert lines[: len(EXPECTED_CMDLINE_ARGS)] == [
        f"OK    {path} ({arg})" for path, arg in sorted(EXPECTED_CMDLINE_ARGS.items())
    ]
    assert lines[-1] == "All config files are valid."


def test_accepts_str_config_dir(config_dir: Path) -> None:
    results = validate_configs(
        Config, CONFIG_GROUPS, schema_name="main_config", config_dir=str(config_dir)
    )
    assert all(r.ok for r in results)


def test_unknown_key(config_dir: Path) -> None:
    (config_dir / "model" / "typo.yaml").write_text(
        "defaults:\n  - fcn\n\nnum_hiden: 3\n"
    )
    results = _validate(config_dir)
    failures = {path for path, r in results.items() if not r.ok}
    assert failures == {"model/typo.yaml"}
    result = results["model/typo.yaml"]
    assert result.cmdline_arg == "model=typo"
    assert isinstance(result.error, ConfigCompositionException)
    assert "num_hiden" in str(result.error)
    assert result.traceback is not None
    assert "num_hiden" in result.traceback


def test_wrong_value_type(config_dir: Path) -> None:
    (config_dir / "model" / "bad.yaml").write_text(
        "defaults:\n  - fcn\n\nnum_hidden: many\n"
    )
    results = _validate(config_dir)
    failures = {path for path, r in results.items() if not r.ok}
    assert failures == {"model/bad.yaml"}
    assert isinstance(
        results["model/bad.yaml"].error, (ValidationError, ConfigCompositionException)
    )


def test_unknown_enum_value(config_dir: Path) -> None:
    (config_dir / "model" / "bad.yaml").write_text(
        "defaults:\n  - cnn\n\nactivation: TANH\n"
    )
    results = _validate(config_dir)
    failures = {path for path, r in results.items() if not r.ok}
    assert failures == {"model/bad.yaml"}
    assert "TANH" in str(results["model/bad.yaml"].error)


def test_missing_parent_config(config_dir: Path) -> None:
    (config_dir / "dm" / "orphan.yaml").write_text("defaults:\n  - does_not_exist\n")
    results = _validate(config_dir)
    failures = {path for path, r in results.items() if not r.ok}
    assert failures == {"dm/orphan.yaml"}
    assert isinstance(results["dm/orphan.yaml"].error, MissingConfigException)


def test_primary_config_without_schema(config_dir: Path) -> None:
    """A primary config that doesn't inherit from the schema isn't type-checked by Hydra."""
    (config_dir / "untyped.yaml").write_text("seed: 1\nnum_hiden: 3\n")
    results = _validate(config_dir)
    failures = {path for path, r in results.items() if not r.ok}
    assert failures == {"untyped.yaml"}
    result = results["untyped.yaml"]
    assert result.cmdline_arg == "--config-name untyped"
    assert isinstance(result.error, TypeError)
    assert "`Config`" in str(result.error)


def test_sweeper_over_nonexistent_key(config_dir: Path) -> None:
    (config_dir / "hydra" / "sweeper" / "bad.yaml").write_text(
        "defaults:\n  - basic\n\nparams:\n  model.hidden_dim: 1,2\n  model.num_layers: 1,2\n"
    )
    results = _validate(config_dir)
    failures = {path for path, r in results.items() if not r.ok}
    assert failures == {"hydra/sweeper/bad.yaml"}
    result = results["hydra/sweeper/bad.yaml"]
    assert result.cmdline_arg == "hydra/sweeper=bad"
    assert isinstance(result.error, KeyError)
    assert "model.num_layers" in str(result.error)


def test_check_configs_reports_failures(config_dir: Path) -> None:
    (config_dir / "model" / "typo.yaml").write_text(
        "defaults:\n  - fcn\n\nnum_hiden: 3\n"
    )
    (config_dir / "dm" / "orphan.yaml").write_text("defaults:\n  - does_not_exist\n")

    out = io.StringIO()
    exit_code = check_configs(
        Config,
        CONFIG_GROUPS,
        schema_name="main_config",
        config_dir=config_dir,
        file=out,
    )
    assert exit_code == 1
    output = out.getvalue()
    assert "FAIL  model/typo.yaml (model=typo)" in output
    assert "FAIL  dm/orphan.yaml (dm=orphan)" in output
    assert "OK    good_run.yaml (--config-name good_run)" in output
    assert "Traceback (most recent call last):" in output
    assert output.rstrip().endswith(
        "2 config file(s) are invalid:\n  - dm/orphan.yaml (dm=orphan)\n  - model/typo.yaml (model=typo)"
    )


@dataclass(kw_only=True)
class SchemaOnlyConfig:
    """A main config class without a defaults list.

    The defaults are instead chosen in a YAML primary config that inherits from the
    schema.
    """

    dm: DataModule
    model: ModelFactory
    opt: OptimisationCfg = field(default_factory=OptimisationCfg)
    seed: int = 42


@pytest.fixture
def split_config_dir(tmp_path: Path) -> Path:
    """A config directory whose default primary config is a YAML file, not the schema."""
    config_dir = tmp_path / "configs"
    (config_dir / "model").mkdir(parents=True)
    (config_dir / "hydra" / "sweeper").mkdir(parents=True)
    (config_dir / "default_config.yaml").write_text(
        "defaults:\n  - base_config\n  - dm: cmnist\n  - model: fcn\n  - opt: null\n"
    )
    (config_dir / "other_run.yaml").write_text(
        "defaults:\n  - base_config\n  - dm: celeba\n  - model: cnn\n\nseed: 1\n"
    )
    (config_dir / "model" / "small.yaml").write_text(
        "defaults:\n  - fcn\n\nnum_hidden: 0\n"
    )
    (config_dir / "hydra" / "sweeper" / "grid.yaml").write_text(
        "defaults:\n  - basic\n\nparams:\n  model.num_hidden: 1,2\n"
    )
    return config_dir


def test_separate_config_with_defaults(split_config_dir: Path) -> None:
    results = validate_configs(
        SchemaOnlyConfig,
        CONFIG_GROUPS,
        schema_name="base_config",
        config_dir=split_config_dir,
        config_with_defaults="default_config",
    )
    assert {r.path.as_posix(): r.cmdline_arg for r in results} == {
        "default_config.yaml": "--config-name default_config",
        "hydra/sweeper/grid.yaml": "hydra/sweeper=grid",
        "model/small.yaml": "model=small",
        "other_run.yaml": "--config-name other_run",
    }
    for result in results:
        assert result.ok, result.traceback


def test_separate_schema_error_message_names_schema(split_config_dir: Path) -> None:
    (split_config_dir / "untyped.yaml").write_text("seed: 1\n")
    results = validate_configs(
        SchemaOnlyConfig,
        CONFIG_GROUPS,
        schema_name="base_config",
        config_dir=split_config_dir,
        config_with_defaults="default_config",
    )
    failures = {r.path.as_posix(): r for r in results if not r.ok}
    assert set(failures) == {"untyped.yaml"}
    assert isinstance(failures["untyped.yaml"].error, TypeError)
    assert "`base_config`" in str(failures["untyped.yaml"].error)


def test_check_configs_passes_config_with_defaults(split_config_dir: Path) -> None:
    out = io.StringIO()
    exit_code = check_configs(
        SchemaOnlyConfig,
        CONFIG_GROUPS,
        schema_name="base_config",
        config_dir=split_config_dir,
        config_with_defaults="default_config",
        file=out,
    )
    assert exit_code == 0, out.getvalue()
    assert out.getvalue().splitlines()[-1] == "All config files are valid."


def test_validate_single_configs(config_dir: Path) -> None:
    (config_dir / "model" / "typo.yaml").write_text(
        "defaults:\n  - fcn\n\nnum_hiden: 3\n"
    )
    with setup_validation(
        Config, CONFIG_GROUPS, schema_name="main_config", config_dir=config_dir
    ) as context:
        assert context.config_dir == config_dir.resolve()
        relative = validate_config(Path("dm") / "celeba_male_smiling.yaml", context)
        absolute = validate_config(
            config_dir / "dm" / "celeba_male_smiling.yaml", context
        )
        broken = validate_config(Path("model") / "typo.yaml", context)
        with pytest.raises(ValueError, match="is not in the subpath"):
            validate_config(config_dir.parent / "elsewhere.yaml", context)

    assert relative == absolute
    assert relative.ok, relative.traceback
    assert relative.path == Path("dm") / "celeba_male_smiling.yaml"
    assert relative.cmdline_arg == "dm=celeba_male_smiling"
    assert not broken.ok
    assert broken.cmdline_arg == "model=typo"
    assert isinstance(broken.error, ConfigCompositionException)


def test_setup_validation_cleans_up(config_dir: Path) -> None:
    with setup_validation(
        Config, CONFIG_GROUPS, schema_name="main_config", config_dir=config_dir
    ):
        assert GlobalHydra.instance().is_initialized()
    assert not GlobalHydra.instance().is_initialized()


def test_malformed_config_class_raises(config_dir: Path) -> None:
    """Problems with the config classes themselves are raised, not recorded."""
    with pytest.raises(ValueError, match="should be a dataclass"):
        validate_configs(int, {}, schema_name="main_config", config_dir=config_dir)


@pytest.mark.parametrize(
    ("name_parts", "groups_in_defaults", "expected"),
    [
        (("dm", "celeba"), {"dm", "model"}, "dm=celeba"),
        (("opt", "adamw"), {"dm", "model"}, "+opt=adamw"),
        (("hydra", "launcher", "slurm"), {"hydra/launcher"}, "hydra/launcher=slurm"),
        (("hydra", "launcher", "slurm"), set(), "+hydra=launcher/slurm"),
        (("experiment", "cmnist", "cnn"), {"dm"}, "+experiment=cmnist/cnn"),
        (("experiment", "cmnist", "cnn"), {"experiment"}, "experiment=cmnist/cnn"),
        (
            ("experiment", "cmnist", "cnn"),
            {"experiment/cmnist"},
            "experiment/cmnist=cnn",
        ),
        (("a", "b", "c", "d"), {"a", "a/b"}, "a/b=c/d"),
    ],
)
def test_construct_override(
    name_parts: tuple[str, ...], groups_in_defaults: set[str], expected: str
) -> None:
    assert construct_override(name_parts, groups_in_defaults) == expected


def test_to_object_removes_hydra_node() -> None:
    cfg = OmegaConf.create(
        {"hydra": {"sweeper": {"params": {"seed": "1,2"}}}, "seed": 1}
    )
    assert _to_object(cfg) == {"seed": 1}

    cfg = OmegaConf.create(
        {"hydra": {"sweeper": {"params": {"nope": "1,2"}}}, "seed": 1}
    )
    with pytest.raises(KeyError, match="nope"):
        _to_object(cfg)

    cfg = OmegaConf.create({"hydra": {"sweeper": {}}, "seed": 1})
    assert _to_object(cfg) == {"seed": 1}
