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

            def ocr_quad(self, img_bgr, quad):
                # ppocr-path stub: crop the quad bbox, delegate to ocr_pil
                q = np.asarray(quad, dtype=np.float32)
                x1 = max(0, int(q[:, 0].min()))
                y1 = max(0, int(q[:, 1].min()))
                x2 = int(q[:, 0].max()) + 1
                y2 = int(q[:, 1].max()) + 1
                crop = img_bgr[y1:y2, x1:x2]
                from PIL import Image as _Image

                i = self.outer._ti
                self.outer._ti += 1
                t = self.outer._texts[i] if i < len(self.outer._texts) else "text"
                return t, q

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

    # Hayai path: min_line_px gates the recognizer (ppocr path has no gate)
    cfg = ComictxtConfig().with_overrides({"rec.backend": "onnx"})
    tiny = np.array([[0, 0], [5, 0], [5, 5], [0, 5]], dtype=np.float32)
    pipe = StubPipeline(cfg, boxes=[[10, 10, 100, 100]], polys_per_region=[[tiny]], texts=["xy"])
    out = pipe.process_pil(_img(ink=False))
    # tiny poly skipped, then fallback rec on white region -> blank gate -> nothing
    assert out["paragraphs"] == []


def test_min_line_px_disabled_at_zero():
    import numpy as np

    cfg = ComictxtConfig().with_overrides(
        {"rec.backend": "onnx", "lines.min_line_px": 0, "rec.blank_std_thresh": 0}
    )
    tiny = np.array([[0, 0], [5, 0], [5, 5], [0, 5]], dtype=np.float32)
    pipe = StubPipeline(cfg, boxes=[[10, 10, 100, 100]], polys_per_region=[[tiny]], texts=["xy"])
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


def test_hayai_line_path_deskews_rotated_quad():
    """Rotated quads must reach the recognizer deskewed, not as axis bboxes.

    A 20x100 line rotated 20deg has a ~53x101 axis bbox; the recognizer
    must see the tight 20x100 warp (plus line_pad border), while the output
    box stays the tight quad xyxy.
    """
    import math

    from PIL import Image, ImageDraw

    cfg = ComictxtConfig().with_overrides(
        {"rec.backend": "onnx",  # Hayai warped-quad path (rec is stubbed)
         "rec.blank_std_thresh": 0, "rec.line_pad": 0, "rec.min_crop_size": 0}
    )
    a = math.radians(20.0)
    c, s = math.cos(a), math.sin(a)
    quad = np.array(
        [[100 + dx * c - dy * s, 100 + dx * s + dy * c]
         for dx, dy in ((-10, -50), (10, -50), (10, 50), (-10, 50))],
        dtype=np.float32,
    )
    seen: dict = {}

    class CapRec:
        def ocr_pil(self, img):
            seen["size"] = img.size
            return "xy"  # 2 chars: single-letter lines hit the junk gate

        def ensure_loaded(self):
            pass

    pipe = StubPipeline(cfg, boxes=[[10, 10, 190, 190]],
                        polys_per_region=[[quad]], texts=[])
    pipe._rec_stub = CapRec()

    @property
    def _rec_prop(self):
        return self._rec_stub

    # swap the rec property on the instance's class for this test only
    # (restore afterwards: delattr would leave later tests falling through
    # to the real lazy-loading ComicTxtPipeline.rec)
    _orig_rec = type(pipe).rec
    type(pipe).rec = _rec_prop
    try:
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        ImageDraw.Draw(img).rectangle([80, 40, 120, 160], fill=(0, 0, 0))
        out = pipe.process_pil(img)
    finally:
        type(pipe).rec = _orig_rec
    assert out["paragraphs"][0]["lines"][0]["text"] == "xy"
    # deskewed crop, not the inflated axis bbox
    assert seen["size"] == (20, 100)


def test_orphan_sweep_recovers_missed_region():
    """Region flow finds one line; the sweep adopts a second, orphan line."""
    region_quad = np.array([[5, 5], [50, 5], [50, 20], [5, 20]], dtype=np.float32)
    orphan_quad = np.array([[120, 120], [160, 120], [160, 140], [120, 140]], dtype=np.float32)

    class SweepStub(StubPipeline):
        @property
        def lines(self):
            outer = self

            class L:
                def __init__(self):
                    self.calls = 0

                def detect_pil(self, img):
                    idx = self.calls
                    self.calls += 1
                    if idx == 0:
                        return [region_quad]
                    return [orphan_quad]  # full-page sweep call

                def ensure_loaded(self):
                    pass

            if not hasattr(self, "_sweep_lines"):
                self._sweep_lines = L()
            return self._sweep_lines

    pipe = SweepStub(ComictxtConfig(), boxes=[[10, 10, 100, 100]], texts=["こんにちは"])
    out = pipe.process_pil(_img())
    texts = [ln["text"] for p in out["paragraphs"] for ln in p["lines"]]
    assert "こんにちは" in texts
    assert any(t.startswith("text") for t in texts)  # orphan recognized


def test_orphan_sweep_disabled():
    region_quad = np.array([[5, 5], [50, 5], [50, 20], [5, 20]], dtype=np.float32)
    orphan_quad = np.array([[120, 120], [160, 120], [160, 140], [120, 140]], dtype=np.float32)

    class SweepStub(StubPipeline):
        @property
        def lines(self):
            outer = self

            class L:
                def __init__(self):
                    self.calls = 0

                def detect_pil(self, img):
                    idx = self.calls
                    self.calls += 1
                    if idx == 0:
                        return [region_quad]
                    return [orphan_quad]

                def ensure_loaded(self):
                    pass

            if not hasattr(self, "_sweep_lines"):
                self._sweep_lines = L()
            return self._sweep_lines

    cfg = ComictxtConfig().with_overrides({"lines.orphan_sweep": False})
    pipe = SweepStub(cfg, boxes=[[10, 10, 100, 100]], texts=["こんにちは"])
    out = pipe.process_pil(_img())
    texts = [ln["text"] for p in out["paragraphs"] for ln in p["lines"]]
    assert texts == ["こんにちは"]


def test_single_letter_lines_gated():
    # single alphanumeric letters (T, キ) are junk that widens paragraph
    # boxes; they are dropped, punctuation-only lines are not
    quads = [
        np.array([[5, 5], [50, 5], [50, 20], [5, 20]], dtype=np.float32),
        np.array([[5, 30], [20, 30], [20, 45], [5, 45]], dtype=np.float32),
    ]
    pipe = StubPipeline(
        ComictxtConfig(),
        boxes=[[10, 10, 100, 100]],
        polys_per_region=[quads],
        texts=["でも", "T"],
    )
    out = pipe.process_pil(_img())
    texts = [ln["text"] for p in out["paragraphs"] for ln in p["lines"]]
    assert "T" not in texts
    assert "でも" in texts
