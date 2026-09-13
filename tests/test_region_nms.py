"""region.nms flag tests (onnx backend, stubbed session, no weights)."""
import numpy as np
from PIL import Image
from types import SimpleNamespace

from comictxt.config import ComictxtConfig
from comictxt.region_yolo import OnnxYoloBackend


def _backend(nms: bool) -> OnnxYoloBackend:
    b = OnnxYoloBackend(
        model_path="/nonexistent/model.onnx",
        imgsz=(64, 64),
        conf=0.1,
        iou=0.7,
        nms=nms,
        contain_action="keep",  # isolate the NMS stage
    )
    # (cx, cy, w, h, conf) in letterboxed px: A/B near-duplicates, C separate.
    pred = np.array([[
        [32, 34, 10, 0],   # cx
        [32, 32, 50, 0],   # cy
        [40, 40, 10, 0],   # w
        [40, 40, 10, 0],   # h
        [0.9, 0.8, 0.85, 0.05],  # conf (last one filtered)
    ]], dtype=np.float32)

    class FakeSess:
        def run(self, *_a, **_k):
            return [pred]

        def get_inputs(self):
            return [SimpleNamespace(name="images")]

    b._sess = FakeSess()
    return b


def _img():
    return Image.new("RGB", (100, 100), (255, 255, 255))


def test_nms_flag_default_off():
    assert ComictxtConfig().region.nms is False


def test_nms_true_suppresses_duplicates():
    info = _backend(nms=True).detect_pil_debug(_img())
    assert info["n_after_nms"] == 2
    assert "nms_skipped" not in info


def test_nms_false_keeps_all():
    info = _backend(nms=False).detect_pil_debug(_img())
    assert info["n_after_nms"] == 3
    assert info.get("nms_skipped") is True
