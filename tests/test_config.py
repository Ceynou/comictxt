import pytest

from comictxt.config import ComictxtConfig, parse_override_value


def test_defaults():
    cfg = ComictxtConfig()
    assert cfg.region.model_size == "x"
    assert cfg.region.backend == "ultralytics"
    assert cfg.region.conf == 0.16
    assert cfg.region.imgsz == 640
    assert cfg.region.iou == 0.7
    assert cfg.region.contain_action == "merge"
    assert cfg.lines.enable_line_stage is True
    assert cfg.lines.thresh == 0.15
    assert cfg.lines.unclip_ratio == 1.4
    assert cfg.lines.det_margin == 16
    assert cfg.lines.det_min_side == 480
    assert cfg.lines.min_short_side == 6
    assert cfg.lines.box_pad == 4.0
    assert cfg.lines.furigana_size_ratio == 0.70
    assert cfg.rec.precision == "fp32"
    assert cfg.rec.backend == "torch"
    assert cfg.rec.ppocr_trim is True
    assert cfg.preprocess.enable is False
    assert cfg.server.port == 7331


def test_with_overrides():
    cfg = ComictxtConfig().with_overrides(
        {"region.conf": 0.35, "lines.enable_line_stage": False, "server.port": 8000}
    )
    assert cfg.region.conf == 0.35
    assert cfg.lines.enable_line_stage is False
    assert cfg.server.port == 8000


def test_with_overrides_unknown_key():
    with pytest.raises(KeyError):
        ComictxtConfig().with_overrides({"nope.key": 1})


def test_from_toml(tmp_path):
    toml = tmp_path / "c.toml"
    toml.write_text('[region]\nmodel_size = "n"\nconf = 0.1\n[server]\nport = 9999\n')
    cfg = ComictxtConfig.from_toml(toml)
    assert cfg.region.model_size == "n"
    assert cfg.region.conf == 0.1
    assert cfg.server.port == 9999
    # unspecified sections keep defaults
    assert cfg.lines.thresh == 0.15


def test_parse_override_value():
    assert parse_override_value("true") is True
    assert parse_override_value("false") is False
    assert parse_override_value("42") == 42
    assert parse_override_value("0.5") == 0.5
    assert parse_override_value("a,b") == ["a", "b"]
    assert parse_override_value("hello") == "hello"


def test_resolved_conf_auto_uses_table():
    cfg = ComictxtConfig()
    cfg.region.model_size = "n"
    cfg.region.conf = -1
    assert cfg.region.resolved_conf() == pytest.approx(0.251)
    cfg.region.conf = 0.5
    assert cfg.region.resolved_conf() == 0.5
