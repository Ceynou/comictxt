"""AnimeText YOLO region detector with ONNXRuntime and Ultralytics backends.

ONNX model output (verified): `output0` shape (B, 5, N) = (cx, cy, w, h, conf)
with boxes in letterboxed-input pixel coordinates and sigmoid already applied.
Single class (`text_block`), so no class scores.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from comictxt.geometry import containment_suppress, merge_contained_boxes, nms

_LETTERBOX_PAD = 114.0


def letterbox(
    img_rgb: np.ndarray, new_w: int, new_h: int
) -> tuple[np.ndarray, float, float, float]:
    """Resize with aspect preserved + gray padding. Returns (padded, ratio, pad_w, pad_h)."""
    h, w = img_rgb.shape[:2]
    ratio = min(new_w / max(1, w), new_h / max(1, h))
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    import cv2

    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = (new_w - nw) / 2.0
    pad_h = (new_h - nh) / 2.0
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    # copyMakeBorder rounding can be off by 1; force exact size via extra pad/crop
    ph, pw = padded.shape[:2]
    if (pw, ph) != (new_w, new_h):
        padded = cv2.resize(padded, (new_w, new_h))
        # recompute effective pad as exact halves (close enough for box rescale)
        pad_w, pad_h = (new_w - nw) / 2.0, (new_h - nh) / 2.0
    return padded, ratio, pad_w, pad_h


def _apply_containment(
    boxes: np.ndarray,
    scores: np.ndarray,
    contain_thresh: float,
    contain_action: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Post-NMS nested-box handling. 'merge' unions contained fragments into
    their container regardless of score order; 'drop' removes fragments
    contained in a higher-scored box; anything else keeps all boxes."""
    if len(boxes) <= 1 or contain_thresh <= 0:
        return boxes, scores
    if contain_action == "merge":
        return merge_contained_boxes(boxes, scores, contain_thresh)
    if contain_action == "drop":
        kept = containment_suppress(boxes, scores, contain_thresh)
        return boxes[kept].astype(np.float32), scores[kept].astype(np.float32)
    return boxes, scores


class OnnxYoloBackend:
    def __init__(
        self,
        model_path: Union[str, Path],
        imgsz: tuple[int, int] = (640, 640),
        conf: float = 0.12,
        iou: float = 0.7,
        nms: bool = False,
        max_det: int = 300,
        providers: Optional[list[str]] = None,
        contain_thresh: float = 0.85,
        contain_action: str = "keep",
    ) -> None:
        self.model_path = Path(model_path)
        self.imgsz = (int(imgsz[0]), int(imgsz[1]))  # (w, h)
        self.conf = float(conf)
        self.iou = float(iou)
        self.nms_enabled = bool(nms)
        self.max_det = int(max_det)
        self.providers = providers or ["CPUExecutionProvider"]
        self.contain_thresh = float(contain_thresh)
        self.contain_action = contain_action
        self._sess = None
        self._input_name = "images"

    def load(self) -> "OnnxYoloBackend":
        import onnxruntime as ort

        if not self.model_path.is_file():
            raise FileNotFoundError(f"YOLO onnx not found: {self.model_path}")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=self.providers
        )
        try:
            self._input_name = self._sess.get_inputs()[0].name
        except Exception:
            self._input_name = "images"
        return self

    def ensure_loaded(self) -> None:
        if self._sess is None:
            self.load()

    def detect_pil(self, img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        info = self.detect_pil_debug(img)
        return info["boxes"], info["scores"]

    def detect_pil_debug(self, img: Image.Image) -> dict:
        """Detect regions with per-step diagnostics (conf/NMS/containment)."""
        self.ensure_loaded()
        assert self._sess is not None
        img = img.convert("RGB")
        W, H = img.size
        arr = np.array(img)  # RGB HWC
        new_w, new_h = self.imgsz
        padded, ratio, pad_w, pad_h = letterbox(arr, new_w, new_h)
        inp = padded.astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[np.newaxis, ...]
        out = self._sess.run(None, {self._input_name: inp})[0]  # (1,5,N)
        pred = out[0].transpose(1, 0)  # (N,5): cx,cy,w,h,conf (letterboxed px)
        info: dict = {
            "backend": "onnx",
            "params": {
                "conf": self.conf,
                "iou": self.iou,
                "nms": self.nms_enabled,
                "max_det": self.max_det,
                "contain_thresh": self.contain_thresh,
                "contain_action": self.contain_action,
            },
            "n_raw": int(len(pred)),
        }
        conf = pred[:, 4]
        keep = conf >= self.conf
        pred = pred[keep]
        conf = conf[keep]
        info["n_after_conf"] = int(len(pred))
        if len(pred) == 0:
            empty = (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32))
            info.update({"boxes_pre_nms": empty[0], "boxes": empty[0], "scores": empty[1]})
            return info
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x1 = (cx - w / 2.0 - pad_w) / ratio
        y1 = (cy - h / 2.0 - pad_h) / ratio
        x2 = (cx + w / 2.0 - pad_w) / ratio
        y2 = (cy + h / 2.0 - pad_h) / ratio
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H)
        # drop degenerate
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, conf = boxes[valid], conf[valid]
        if len(boxes) == 0:
            info.update({"boxes_pre_nms": boxes, "boxes": boxes, "scores": conf})
            return info
        order = np.argsort(-conf)[: self.max_det * 4]  # pre-trim before NMS
        boxes, conf = boxes[order], conf[order]
        info["boxes_pre_nms"] = boxes.astype(np.float32)
        info["scores_pre_nms"] = conf.astype(np.float32)
        if self.nms_enabled:
            kept = nms(boxes, conf, self.iou)[: self.max_det]
            boxes, conf = boxes[kept].astype(np.float32), conf[kept].astype(np.float32)
            info["n_after_nms"] = int(len(boxes))
        else:
            # Merge-only dedup (ablation winner): score-order cap, no IoU-NMS.
            boxes = boxes[: self.max_det].astype(np.float32)
            conf = conf[: self.max_det].astype(np.float32)
            info["n_after_nms"] = int(len(boxes))
            info["nms_skipped"] = True
        boxes, conf = _apply_containment(boxes, conf, self.contain_thresh, self.contain_action)
        info["n_after_containment"] = int(len(boxes))
        info.update({"boxes": boxes, "scores": conf})
        return info


class UltralyticsYoloBackend:
    def __init__(
        self,
        model_path: Union[str, Path],
        imgsz: tuple[int, int] = (640, 640),
        conf: float = 0.12,
        iou: float = 0.7,
        max_det: int = 300,
        device: str = "cpu",
        contain_thresh: float = 0.85,
        contain_action: str = "keep",
    ) -> None:
        self.model_path = Path(model_path)
        self.imgsz = (int(imgsz[0]), int(imgsz[1]))
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.device = device
        self.contain_thresh = float(contain_thresh)
        self.contain_action = contain_action
        self._model = None

    def load(self) -> "UltralyticsYoloBackend":
        import os

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "Ultralytics backend requires `pip install ultralytics torch`. "
                "Use backend='onnx' otherwise."
            ) from e
        if not self.model_path.is_file():
            raise FileNotFoundError(f"YOLO .pt not found: {self.model_path}")
        if any(
            os.environ.get(v, "").strip().lower() in ("1", "true", "yes")
            for v in ("HF_HUB_OFFLINE", "COMICTXT_OFFLINE")
        ):
            # Keep ultralytics from phoning home (update checks) when offline;
            # model file itself is local so inference needs no network.
            os.environ.setdefault("YOLO_VERBOSE", "False")
            os.environ.setdefault("ULTRALYTICS_HIDE_UPDATE_MSG", "True")
        try:
            self._model = YOLO(str(self.model_path))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO weights from {self.model_path}. If you are "
                f"offline, ensure the local .pt exists and set region.pt_path "
                f"explicitly. Original error: {e}"
            ) from e
        return self

    def ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    def detect_pil(self, img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        info = self.detect_pil_debug(img)
        return info["boxes"], info["scores"]

    def detect_pil_debug(self, img: Image.Image) -> dict:
        """Detect regions. NMS runs inside ultralytics; only final boxes
        (post-NMS + containment) are observable — raw proposals are not."""
        self.ensure_loaded()
        assert self._model is not None
        img = img.convert("RGB")
        W, H = img.size
        imgsz_arg = max(self.imgsz)  # ultralytics takes square int
        results = self._model.predict(
            img,
            imgsz=imgsz_arg,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        info: dict = {
            "backend": "ultralytics",
            "params": {
                "conf": self.conf,
                "iou": self.iou,
                "max_det": self.max_det,
                "contain_thresh": self.contain_thresh,
                "contain_action": self.contain_action,
            },
            "note": "NMS is internal to ultralytics (iou param); "
            "pre-NMS proposals are not observable via this backend. "
            "Use region.backend='onnx' for full conf/NMS/containment steps.",
        }
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            empty = (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32))
            info.update(
                {
                    "n_raw": 0,
                    "n_after_nms": 0,
                    "n_after_containment": 0,
                    "boxes": empty[0],
                    "scores": empty[1],
                }
            )
            return info
        data = results[0].boxes.data.cpu().numpy()  # (N,6): xyxy+conf+cls
        boxes = data[:, :4].astype(np.float32)
        scores = data[:, 4].astype(np.float32)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H)
        info["n_after_nms"] = int(len(boxes))
        boxes, scores = _apply_containment(boxes, scores, self.contain_thresh, self.contain_action)
        info["n_after_containment"] = int(len(boxes))
        info.update({"boxes": boxes, "scores": scores})
        return info


class RegionDetector:
    """Unified facade selecting the configured backend."""

    def __init__(
        self,
        backend: str = "onnx",
        model_size: str = "x",
        onnx_path: str = "",
        pt_path: str = "",
        imgsz: tuple[int, int] = (640, 640),
        conf: float = 0.12,
        iou: float = 0.7,
        nms: bool = False,
        max_det: int = 300,
        providers: Optional[list[str]] = None,
        device: str = "cpu",
        contain_thresh: float = 0.85,
        contain_action: str = "keep",
    ) -> None:
        if backend not in ("onnx", "ultralytics"):
            raise ValueError("region backend must be 'onnx' or 'ultralytics'")
        if contain_action not in ("drop", "keep", "merge"):
            raise ValueError("region contain_action must be 'drop', 'keep' or 'merge'")
        self.backend_name = backend
        self._lock = threading.Lock()  # ultralytics predict is not documented thread-safe
        self._impl: Union[OnnxYoloBackend, UltralyticsYoloBackend]
        if backend == "onnx":
            from comictxt.config import resolve_region_onnx

            self._impl = OnnxYoloBackend(
                resolve_region_onnx(model_size, onnx_path),
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                nms=nms,
                max_det=max_det,
                providers=providers,
                contain_thresh=contain_thresh,
                contain_action=contain_action,
            )
        else:
            from comictxt.config import resolve_region_pt

            self._impl = UltralyticsYoloBackend(
                resolve_region_pt(model_size, pt_path),
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                max_det=max_det,
                device=device,
                contain_thresh=contain_thresh,
                contain_action=contain_action,
            )

    def load(self) -> "RegionDetector":
        self._impl.load()
        return self

    def ensure_loaded(self) -> None:
        self._impl.ensure_loaded()

    def detect_pil(self, img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._impl.detect_pil(img)

    def detect_pil_debug(self, img: Image.Image) -> dict:
        """Like detect_pil but with per-step diagnostics (see backends)."""
        with self._lock:
            debug = getattr(self._impl, "detect_pil_debug", None)
            if debug is None:
                boxes, scores = self._impl.detect_pil(img)
                return {"backend": "unknown", "boxes": boxes, "scores": scores}
            return debug(img)
