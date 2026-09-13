"""Parallelism tests with stateless stubs (no model weights)."""
import numpy as np
from PIL import Image

from comictxt.cli import _safe_process
from comictxt.config import ComictxtConfig
from comictxt.pipeline import ComicTxtPipeline


class StatelessStub(ComicTxtPipeline):
    """Fixed regions; rec text derived from crop size (deterministic, stateless)."""

    BOXES = [[10, 10, 60, 60], [70, 70, 130, 130], [140, 140, 220, 220]]

    @property
    def region(self):
        class R:
            def detect_pil(self, img):
                b = np.asarray(StatelessStub.BOXES, dtype=np.float32)
                return b, np.ones((len(b),), dtype=np.float32)

            def ensure_loaded(self):
                pass

        return R()

    @property
    def lines(self):
        class L:
            def detect_pil(self, img):
                w, h = img.size
                return [np.array([[0, 0], [w - 1, 0], [w - 1, 9], [0, 9]], dtype=np.float32)]

            def ensure_loaded(self):
                pass

        return L()

    @property
    def rec(self):
        class R:
            def ocr_pil(self, img):
                return f"w{img.size[0]}"

            def ensure_loaded(self):
                pass

        return R()


def _img():
    from PIL import ImageDraw

    img = Image.new("RGB", (300, 300), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for x in range(0, 300, 10):  # stripes: every crop has variance (passes blank gate)
        d.rectangle([x, 0, x + 4, 300], fill=(0, 0, 0))
    return img


def test_inner_fanout_preserves_region_order():
    cfg = ComictxtConfig()  # workers auto -> inner parallel on (3 boxes)
    out = StatelessStub(cfg).process_pil(_img())
    assert len(out["paragraphs"]) == 3
    texts = [p["lines"][0]["text"] for p in out["paragraphs"]]
    assert texts == sorted(texts)  # crop widths grow with region order
    out_seq = StatelessStub(cfg, allow_inner_parallel=False).process_pil(_img())
    assert [p["lines"][0]["text"] for p in out_seq["paragraphs"]] == texts


def test_safe_process_isolates_errors(tmp_path):
    from pathlib import Path

    class Boom(StatelessStub):
        def process_image(self, image):
            raise RuntimeError("boom")

    p = Path("x.png")
    res = _safe_process(Boom(ComictxtConfig()), p)
    assert isinstance(res, RuntimeError)


def test_batch_threads_order_and_isolation(tmp_path, monkeypatch):
    import comictxt.pipeline as pipe_mod
    from comictxt.cli import _infer_batch_threads

    for i in range(3):
        _img().save(tmp_path / f"{i}.png")
    imgs = [tmp_path / f"{i}.png" for i in range(3)]
    monkeypatch.setattr(pipe_mod, "ComicTxtPipeline", StatelessStub)
    results = _infer_batch_threads(ComictxtConfig(), imgs, workers=2)
    assert [p.name for p, _ in results] == ["0.png", "1.png", "2.png"]
    for _, res in results:
        assert not isinstance(res, Exception)
        assert len(res["paragraphs"]) == 3
