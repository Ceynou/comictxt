"""Pipeline tests with stubbed detectors (no model weights needed)."""
import numpy as np
from PIL import Image

from comictxt.config import ComictxtConfig
from comictxt.pipeline import ComicTxtPipeline


class StubPipeline(ComicTxtPipeline):
    """Bypass lazy model loading; inject canned detections."""

    def __init__(self, cfg, boxes, polys_per_region=None, texts=None):
        super().__init__(cfg)
        self._boxes = np.asarray(boxes, dtype=np.float32)
        self._polys = polys_per_region or []
        self._texts = texts or []
        self._ti = 0

    @property
    def region(self):
        class R:
            def __init__(self, outer):
                self.outer = outer

            def detect_pil(self, img):
                scores = np.ones((len(self.outer._boxes),), dtype=np.float32)
                return self.outer._boxes, scores

            def ensure_loaded(self):
                pass

        return R(self)

    @property
    def lines(self):
        class L:
            def __init__(self, outer):
                self.outer = outer
                self.calls = 0

            def detect_pil(self, img):
                # return canned polys for region index == calls
                idx = self.calls
                self.calls += 1
                if idx < len(self.outer._polys):
                    return self.outer._polys[idx]
                return []

            def ensure_loaded(self):
                pass

        if not hasattr(self, "_stub_lines"):
            self._stub_lines = L(self)
        return self._stub_lines

    @property
    def rec(self):
        class R:
            def __init__(self, outer):
                self.outer = outer

            def ocr_pil(self, img):
                i = self.outer._ti
                self.outer._ti += 1
                if i < len(self.outer._texts):
                    return self.outer._texts[i]
                return "text"

            def ensure_loaded(self):
                pass

        return R(self)


def _img(ink=True):
    from PIL import ImageDraw

    img = Image.new("RGB", (200, 200), (255, 255, 255))
    if ink:
        d = ImageDraw.Draw(img)
        for x in range(0, 200, 10):  # stripes: passes blank gate
            d.rectangle([x, 0, x + 4, 200], fill=(0, 0, 0))
    return img


def test_region_lines_rec_flow():
    cfg = ComictxtConfig()
    pipe = StubPipeline(
        cfg,
        boxes=[[10, 10, 100, 100]],
        polys_per_region=[[np.array([[5, 5], [50, 5], [50, 20], [5, 20]], dtype=np.float32)]],
        texts=["こんにちは"],
    )
    out = pipe.process_pil(_img())
    assert out["image_properties"] == {"width": 200, "height": 200}
    assert len(out["paragraphs"]) == 1
    assert out["paragraphs"][0]["lines"][0]["text"] == "こんにちは"


def test_direct_region_mode_when_lines_disabled():
    cfg = ComictxtConfig().with_overrides({"lines.enable_line_stage": False})
    pipe = StubPipeline(cfg, boxes=[[10, 10, 50, 50]], texts=["hello"])
    out = pipe.process_pil(_img())
    assert len(out["paragraphs"]) == 1
    assert out["paragraphs"][0]["lines"][0]["text"] == "hello"


def test_fallback_to_region_when_no_lines():
    cfg = ComictxtConfig()  # fallback_to_region_text=True by default
    pipe = StubPipeline(cfg, boxes=[[10, 10, 50, 50]], polys_per_region=[[]], texts=["fallback"])
    out = pipe.process_pil(_img())
    assert len(out["paragraphs"]) == 1
    assert out["paragraphs"][0]["lines"][0]["text"] == "fallback"


def test_no_fallback_means_no_paragraph():
    cfg = ComictxtConfig().with_overrides({"lines.fallback_to_region_text": False})
    pipe = StubPipeline(cfg, boxes=[[10, 10, 50, 50]], polys_per_region=[[]], texts=[])
    out = pipe.process_pil(_img())
    assert out["paragraphs"] == []


def test_empty_text_lines_skipped():
    cfg = ComictxtConfig().with_overrides({"lines.enable_line_stage": False})
    pipe = StubPipeline(cfg, boxes=[[10, 10, 50, 50]], texts=["   "])
    out = pipe.process_pil(_img())
    assert out["paragraphs"] == []


def test_owocr_shape_compatible_with_neokuro_converter():
    cfg = ComictxtConfig()
    pipe = StubPipeline(cfg, boxes=[[10, 10, 100, 60]], texts=["x"])
    out = pipe.process_pil(_img())
    # minimal assertions mirroring neokuro/converter.py::parse_owocr_file access pattern
    assert "image_properties" in out and "paragraphs" in out
    for para in out["paragraphs"]:
        assert "writing_direction" in para
        for line in para["lines"]:
            bb = line["bounding_box"]
            for k in ("center_x", "center_y", "width", "height"):
                assert k in bb


def test_min_line_px_skips_specks():
    import numpy as np

    cfg = ComictxtConfig()  # min_line_px=12 default
    tiny = np.array([[0, 0], [5, 0], [5, 5], [0, 5]], dtype=np.float32)
    pipe = StubPipeline(cfg, boxes=[[10, 10, 100, 100]], polys_per_region=[[tiny]], texts=["x"])
    out = pipe.process_pil(_img(ink=False))
    # tiny poly skipped, then fallback rec on white region -> blank gate -> nothing
    assert out["paragraphs"] == []


def test_min_line_px_disabled_at_zero():
    import numpy as np

    cfg = ComictxtConfig().with_overrides({"lines.min_line_px": 0, "rec.blank_std_thresh": 0})
    tiny = np.array([[0, 0], [5, 0], [5, 5], [0, 5]], dtype=np.float32)
    pipe = StubPipeline(cfg, boxes=[[10, 10, 100, 100]], polys_per_region=[[tiny]], texts=["x"])
    out = pipe.process_pil(_img())
    assert len(out["paragraphs"]) == 1


def test_blank_crop_skipped_before_recognizer():
    from PIL import Image

    cfg = ComictxtConfig().with_overrides({"lines.enable_line_stage": False})
    pipe = StubPipeline(cfg, boxes=[[10, 10, 50, 50]], texts=["should-not-appear"])
    white = Image.new("RGB", (200, 200), (255, 255, 255))
    out = pipe.process_pil(white)
    assert out["paragraphs"] == []


def test_nonblank_crop_recognized():
    from PIL import Image, ImageDraw

    cfg = ComictxtConfig().with_overrides({"lines.enable_line_stage": False})
    pipe = StubPipeline(cfg, boxes=[[10, 10, 50, 50]], texts=["hello"])
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([10, 10, 50, 50], fill=(0, 0, 0))
    out = pipe.process_pil(img)
    assert out["paragraphs"][0]["lines"][0]["text"] == "hello"
