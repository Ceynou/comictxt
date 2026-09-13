from comictxt.cli import build_parser, _build_config


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
