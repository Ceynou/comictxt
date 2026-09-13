"""Debug report tests with stubbed detectors (no model weights needed)."""
import numpy as np
from PIL import Image, ImageDraw

from comictxt.config import ComictxtConfig
from comictxt.debug import run_debug
from comictxt.pipeline import ComicTxtPipeline


class StubPipe(ComicTxtPipeline):
    """Stub region/lines/rec; no *_debug hooks -> exercises fallbacks."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._texts = ["漢字テスト", "ふりがな"]

    @property
    def region(self):
        outer = self

        class R:
            def detect_pil(self, img):
                return (np.array([[10, 10, 100, 120]], dtype=np.float32),
                        np.array([0.9], dtype=np.float32))

            def ensure_loaded(self):
                pass

        return R()

    @property
    def lines(self):
        class L:
            def detect_pil(self, img):
                # full-image coords; main kanji line + small kana line beside it
                return [np.array([[20, 20], [60, 20], [60, 100], [20, 100]],
                                 dtype=np.float32)]

            def ensure_loaded(self):
                pass

        return L()

    @property
    def rec(self):
        outer = self

        class R:
            def ocr_pil(self, img):
                return outer._texts.pop(0) if outer._texts else "text"

            def ensure_loaded(self):
                pass

        return R()


def _img():
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 100, 120], fill=(0, 0, 0))
    return img


def test_debug_report_writes_html_images_json(tmp_path):
    cfg = ComictxtConfig().with_overrides({"rec.blank_std_thresh": 0})
    pipe = StubPipe(cfg)
    index = run_debug(_img(), tmp_path / "dbg", pipe=pipe)
    assert index.is_file()
    content = index.read_text(encoding="utf-8")
    assert "Region detection" in content
    assert "furigana" in content
    assert (tmp_path / "dbg" / "stages" / "00_input.png").is_file()
    assert (tmp_path / "dbg" / "stages" / "01_regions.png").is_file()
    assert (tmp_path / "dbg" / "stages" / "01_regions.json").is_file()
    assert (tmp_path / "dbg" / "stages" / "02_padded.png").is_file()
    assert (tmp_path / "dbg" / "stages" / "05_final.png").is_file()
    assert (tmp_path / "dbg" / "result.json").is_file()


def test_debug_report_no_regions(tmp_path):
    cfg = ComictxtConfig().with_overrides({"rec.blank_std_thresh": 0})

    class EmptyRegion(StubPipe):
        @property
        def region(self):
            class R:
                def detect_pil(self, img):
                    return (np.zeros((0, 4), dtype=np.float32),
                            np.zeros((0,), dtype=np.float32))

                def ensure_loaded(self):
                    pass

            return R()

    index = run_debug(_img(), tmp_path / "dbg", pipe=EmptyRegion(cfg))
    assert index.is_file()
