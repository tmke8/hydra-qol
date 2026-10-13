from pathlib import Path
import shutil

import pytest

CONFIG_DIR = Path(__file__).parent / "configs"


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A writable copy of the valid example config directory."""
    dest = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, dest)
    return dest
