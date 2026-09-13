"""Keep tests hermetic: isolate user-config lookup from the real home dir.

Without this, tests calling `_build_config` would read (or, since
first-start auto-create, write) `~/.config/comictxt/config.toml`.
"""
import pytest


@pytest.fixture(autouse=True)
def _isolated_user_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("COMICTXT_CONFIG", raising=False)
