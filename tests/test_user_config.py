"""First-start user-config auto-create policy tests."""
import pytest

from comictxt.cli import _cmd_config, _load_effective_config, build_parser, _build_config
from comictxt.config import ensure_user_config, user_config_path


def _infer_args(*extra):
    return build_parser().parse_args(["infer", "x.png", *extra])


def test_first_start_creates_and_reads_user_config():
    args = _infer_args()
    dest = user_config_path()
    assert not dest.is_file()
    cfg = _build_config(args)
    assert dest.is_file()
    assert "auto-generated on first run" in dest.read_text(encoding="utf-8")
    # created file is honored: edit survives the next start (no overwrite)
    dest.write_text('[server]\nport = 9999\n', encoding="utf-8")
    cfg2 = _build_config(_infer_args())
    assert cfg2.server.port == 9999
    assert cfg.server.port != 9999  # first run used shipped defaults


def test_set_overrides_autocreated_file():
    _build_config(_infer_args())
    cfg = _build_config(_infer_args("--set", "server.port=8888"))
    assert cfg.server.port == 8888


def test_unwritable_home_warns_and_uses_defaults(caplog):
    import os

    ro = user_config_path().parent.parent
    ro.mkdir(parents=True, exist_ok=True)
    os.chmod(ro, 0o555)
    try:
        with caplog.at_level("WARNING", logger="comictxt"):
            cfg = _build_config(_infer_args())
        assert not user_config_path().is_file()
        assert cfg.server.port == 7331  # built-in default
        assert any("Could not write user config" in r.message for r in caplog.records)
    finally:
        os.chmod(ro, 0o755)


def test_config_print_and_path_do_not_create():
    dest = user_config_path()
    assert _cmd_config(build_parser().parse_args(["config", "--print"])) == 0
    assert _cmd_config(build_parser().parse_args(["config", "--path"])) == 0
    assert not dest.is_file()


def test_missing_explicit_config_still_errors():
    args = build_parser().parse_args(
        ["infer", "x.png", "--config", "/nonexistent/c.toml"])
    with pytest.raises(SystemExit):
        _build_config(args)
    assert not user_config_path().is_file()


def test_ensure_user_config_never_overwrites():
    first = ensure_user_config()
    assert first is not None and first.is_file()
    first.write_text("[server]\nport = 1234\n", encoding="utf-8")
    second = ensure_user_config()
    assert second == first
    assert "port = 1234" in first.read_text(encoding="utf-8")


def test_load_effective_config_without_create():
    args = _infer_args()
    cfg = _load_effective_config(args, create_user_config=False)
    assert cfg.server.port == 7331
    assert not user_config_path().is_file()
