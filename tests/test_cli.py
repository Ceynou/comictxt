from comictxt.cli import _cmd_config, build_parser, _build_config
from comictxt.config import user_config_path


def test_set_before_subcommand():
    args = build_parser().parse_args(["--set", "region.conf=0.35", "infer", "x.png"])
    cfg = _build_config(args)
    assert cfg.region.conf == 0.35


def test_set_after_subcommand():
    args = build_parser().parse_args(["infer", "x.png", "--set", "region.conf=0.35"])
    cfg = _build_config(args)
    assert cfg.region.conf == 0.35


def test_config_after_subcommand(tmp_path):
    toml = tmp_path / "c.toml"
    toml.write_text('[server]\nport = 9999\n')
    args = build_parser().parse_args(["serve", "--config", str(toml)])
    cfg = _build_config(args)
    assert cfg.server.port == 9999


def test_shortcuts():
    args = build_parser().parse_args(["infer", "x.png", "--region-conf", "0.2", "--no-lines"])
    cfg = _build_config(args)
    assert cfg.region.conf == 0.2
    assert cfg.lines.enable_line_stage is False


def test_config_bare_shows_labeled_paths_and_active_user_config(capsys):
    dest = user_config_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("[server]\nport = 9999\n", encoding="utf-8")
    assert _cmd_config(build_parser().parse_args(["config"])) == 0
    out = capsys.readouterr().out
    assert "default:" in out and "user:" in out and "effective:" in out
    assert "user config" in out
    assert "port = 9999" in out  # active custom config is printed
    assert dest.is_file()  # never deleted


def test_config_effective_honors_set_without_writing(capsys):
    dest = user_config_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("[server]\nport = 9999\n", encoding="utf-8")
    args = build_parser().parse_args(
        ["config", "--effective", "--set", "server.port=8888"])
    assert _cmd_config(args) == 0
    out = capsys.readouterr().out
    assert "port = 8888" in out
    assert "port = 9999" not in out
    assert "port = 9999" in dest.read_text(encoding="utf-8")


def test_config_path_is_labeled_and_creates_nothing(capsys):
    dest = user_config_path()
    assert _cmd_config(build_parser().parse_args(["config", "--path"])) == 0
    out = capsys.readouterr().out
    assert "default:" in out and "user:" in out and "effective:" in out
    assert not dest.is_file()
