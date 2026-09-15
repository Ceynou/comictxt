"""Tests for det margin, empty-box drops, config CLI, offline flags."""
import numpy as np
from PIL import Image, ImageDraw

from comictxt.cli import build_parser, _build_config
from comictxt.config import (
    ComictxtConfig,
    is_offline,
    resolve_config_file,
    user_config_path,
)


class StubPipe:
    pass


def _stub_pipeline(cfg, texts):
    from comictxt.pipeline import ComicTxtPipeline

    class P(ComicTxtPipeline):
        def __init__(self):
            super().__init__(cfg)
            self._texts = list(texts)

        @property
        def region(self):
            class R:
                def detect_pil(self, img):
                    return (np.array([[10, 10, 100, 100]], dtype=np.float32),
                            np.array([0.9], dtype=np.float32))

                def ensure_loaded(self):
                    pass

            return R()

        @property
        def lines(self):
            class L:
                def detect_pil(self, img):
                    return [np.array([[5, 5], [50, 5], [50, 20], [5, 20]],
                                     dtype=np.float32)]

                def ensure_loaded(self):
                    pass

            return L()

        @property
        def rec(self):
            outer = self

            class R:
                def ocr_pil(self, img):
                    return outer._texts.pop(0) if outer._texts else ""

                def ensure_loaded(self):
                    pass

            return R()

    return P()


def _img():
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for x in range(0, 200, 10):
        d.rectangle([x, 0, x + 4, 200], fill=(0, 0, 0))
    return img


def test_det_margin_defaults():
    cfg = ComictxtConfig()
    assert cfg.lines.det_margin == 16
    assert cfg.lines.det_min_side == 480
    assert cfg.lines.det_long_side == 960
    assert not hasattr(cfg.lines, "line_expand_ratio")


def test_preprocess_scale_policy_no_weights():
    # _preprocess needs no model session: margin pad + cap/floor + snap32.
    from comictxt.lines_ppocr import LineDetector

    det = LineDetector(model_path="/nonexistent/model.onnx", det_margin=16,
                       det_min_side=480)
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    inp, W, H, pW, pH, tw, th = det._preprocess(arr)
    assert (W, H) == (200, 100)
    assert (pW, pH) == (232, 132)  # +16px white border each side
    # long side 232 < 480 -> upscale to 480, snapped to 32
    assert max(tw, th) == 480
    assert tw % 32 == 0 and th % 32 == 0
    assert inp.shape == (1, 3, th, tw)


def test_preprocess_large_image_capped():
    from comictxt.lines_ppocr import LineDetector

    det = LineDetector(model_path="/nonexistent/model.onnx", det_margin=0)
    arr = np.zeros((1500, 800, 3), dtype=np.uint8)
    inp, W, H, pW, pH, tw, th = det._preprocess(arr)
    assert max(tw, th) == 960  # capped at det_long_side
    assert tw % 32 == 0 and th % 32 == 0


def test_empty_ocr_text_drops_paragraph():
    cfg = ComictxtConfig().with_overrides({"lines.enable_line_stage": False})
    pipe = _stub_pipeline(cfg, ["   "])
    out = pipe.process_pil(_img())
    assert out["paragraphs"] == []
    pipe2 = _stub_pipeline(cfg, [""])
    assert pipe2.process_pil(_img())["paragraphs"] == []


def test_finalize_drops_empty_and_reorders():
    from comictxt.pipeline import ComicTxtPipeline

    cfg = ComictxtConfig()
    pipe = ComicTxtPipeline(cfg)
    good = {"bounding_box": {"center_x": 0.5, "center_y": 0.2, "width": 0.1,
                             "height": 0.1, "rotation_z": 0.0},
            "lines": [{"text": "a", "bounding_box": {"center_x": 0.5, "center_y": 0.2,
                                                     "width": 0.05, "height": 0.05}}],
            "writing_direction": "LEFT_TO_RIGHT"}
    empty = {"bounding_box": {"center_x": 0.5, "center_y": 0.8, "width": 0.1,
                              "height": 0.1, "rotation_z": 0.0},
             "lines": [{"text": "   ", "bounding_box": {}}],
             "writing_direction": "LEFT_TO_RIGHT"}
    out = pipe._finalize_paragraphs([empty, good])
    assert out == [good]


def test_cli_reorder_flags():
    args = build_parser().parse_args(["infer", "x.png", "--no-reorder"])
    assert _build_config(args).pipeline.reorder_blocks is False
    args = build_parser().parse_args(["infer", "x.png", "--no-reorder-blocks"])
    cfg = _build_config(args)
    assert cfg.pipeline.reorder_blocks is False
    assert cfg.pipeline.reorder_lines is True


def test_cli_offline_and_ppocr_flags():
    args = build_parser().parse_args(
        ["infer", "x.png", "--offline", "--rec-backend", "ppocr"])
    cfg = _build_config(args)
    assert cfg.general.offline is True
    assert cfg.rec.backend == "ppocr"


def test_cli_config_subcommand_paths_and_init(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("COMICTXT_CONFIG", raising=False)
    dest = user_config_path()
    assert str(tmp_path) in str(dest)
    args = build_parser().parse_args(["config", "--init"])
    from comictxt.cli import _cmd_config

    assert _cmd_config(args) == 0
    assert dest.is_file()
    # second init without --force refuses
    assert _cmd_config(args) == 1
    # user config is picked up automatically
    cfg_args = build_parser().parse_args(["infer", "x.png"])
    assert isinstance(_build_config(cfg_args), ComictxtConfig)


def test_resolve_config_file_prefers_explicit(tmp_path):
    toml = tmp_path / "c.toml"
    toml.write_text('[server]\nport = 9999\n')
    assert resolve_config_file(toml) == toml


def test_is_offline_env(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    assert is_offline(ComictxtConfig()) is False
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert is_offline(ComictxtConfig()) is True
    cfg = ComictxtConfig().with_overrides({"general.offline": True})
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert is_offline(cfg) is True
