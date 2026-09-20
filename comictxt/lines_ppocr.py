"""PP-OCRv6 manga text-line detector (DB head, ONNX).

Port of the detection half of Kellenok's PP-OCR_manga Space app
(``Kellenok/PP-OCRv6_manga`` Spaces ``app.py::run_ocr``): white-margin pad,
scale policy (cap long side, floor short inputs), threshold + contour +
box-score + pyclipper unclip (largest path wins), ``min_short_side``
minimum (18px — ruby-exclusion knee from the line ablation), then
``minAreaRect + box_pad`` 4-point quads.

Returns quads in input-image pixel coordinates.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import pyclipper
from PIL import Image

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def box_score_fast(pred: np.ndarray, box: np.ndarray) -> float:
    """Mean prob-map value inside the contour polygon (Space verbatim)."""
    h, w = pred.shape[:2]
    b = box.copy().astype(np.int32)
    xmin = np.clip(np.floor(b[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(b[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(b[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(b[:, 1].max()).astype(np.int32), 0, h - 1)
    if xmax < xmin or ymax < ymin:
        return 0.0
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    bb = b.copy()
    bb[:, 0] -= xmin
    bb[:, 1] -= ymin
    cv2.fillPoly(mask, [bb.reshape(-1, 2)], 1)
    return float(cv2.mean(pred[ymin : ymax + 1, xmin : xmax + 1], mask)[0])


def _polygon_area(box: np.ndarray) -> float:
    x, y = box[:, 0], box[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _polygon_length(box: np.ndarray) -> float:
    return float(np.linalg.norm(np.roll(box, -1, axis=0) - box, axis=1).sum())


def unclip(box: np.ndarray, ratio: float) -> Optional[np.ndarray]:
    """Expand a contour by ``area * ratio / perimeter`` (Space ``unclip_pp``).

    Shoelace area/perimeter instead of shapely (no extra dependency).
    Returns the largest output path, or None when pyclipper yields nothing.
    """
    pts = np.asarray(box, dtype=np.float64)
    area = _polygon_area(pts)
    length = _polygon_length(pts)
    if area <= 0 or length <= 0:
        return None
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(pts.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    paths = offset.Execute(area * float(ratio) / length)
    if not paths:
        return None
    paths = sorted(
        paths,
        key=lambda p: _polygon_area(np.asarray(p, dtype=np.float64)) if len(p) >= 3 else 0.0,
        reverse=True,
    )
    if len(paths[0]) == 0:
        return None
    return np.asarray(paths[0], dtype=np.float32)


class LineDetector:
    """DB line detector. Returns 4-point quads in input-image pixel coordinates."""

    def __init__(
        self,
        model_path: Union[str, Path],
        det_long_side: int = 960,
        det_min_side: int = 480,
        det_margin: int = 16,
        thresh: float = 0.15,
        box_thresh: float = 0.25,
        unclip_ratio: float = 1.4,
        min_short_side: int = 6,
        box_pad: float = 4.0,
        max_candidates: int = 3000,
        providers: Optional[list[str]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.det_long_side = int(det_long_side)
        self.det_min_side = int(det_min_side)
        self.det_margin = int(det_margin)
        self.thresh = float(thresh)
        self.box_thresh = float(box_thresh)
        self.unclip_ratio = float(unclip_ratio)
        self.min_short_side = int(min_short_side)
        self.box_pad = float(box_pad)
        self.max_candidates = int(max_candidates)
        self.providers = providers or ["CPUExecutionProvider"]
        self._sess = None
        self._input_name = "x"

    def load(self) -> "LineDetector":
        import onnxruntime as ort

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Line det model not found: {self.model_path}")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=self.providers
        )
        try:
            self._input_name = self._sess.get_inputs()[0].name
        except Exception:
            self._input_name = "x"
        return self

    def ensure_loaded(self) -> None:
        if self._sess is None:
            self.load()

    # -- preprocessing (Space: white margin pad + cap/floor scale policy) ---
    def _preprocess(
        self, img_bgr: np.ndarray
    ) -> tuple[np.ndarray, int, int, int, int, int, int]:
        """Returns (inp, W, H, pW, pH, tw, th).

        The image is padded with a white ``det_margin`` border (so text
        touching the crop edge still detects), then scaled: long side capped
        at ``det_long_side``, inputs smaller than ``det_min_side`` upscaled to
        it, snapped to multiples of 32 (min 64). Mirrors the Space app.
        """
        H, W = img_bgr.shape[:2]
        margin = max(0, self.det_margin)
        if margin > 0:
            pad_img = cv2.copyMakeBorder(
                img_bgr, margin, margin, margin, margin,
                cv2.BORDER_CONSTANT, value=[255, 255, 255],
            )
        else:
            pad_img = img_bgr
        pH, pW = pad_img.shape[:2]
        target_max = max(pH, pW)
        if target_max > self.det_long_side:
            scale = float(self.det_long_side) / target_max
        elif target_max < self.det_min_side:
            scale = max(1.0, float(self.det_min_side) / target_max)
        else:
            scale = 1.0
        th = max(int(round(pH * scale / 32) * 32), 64)
        tw = max(int(round(pW * scale / 32) * 32), 64)
        resized = cv2.resize(pad_img, (tw, th)).astype(np.float32) / 255.0
        # NOTE: Space feeds BGR pixels with RGB mean/std (Paddle convention);
        # do NOT swap channels.
        normed = (resized - _MEAN) / _STD
        inp = normed.transpose((2, 0, 1))[np.newaxis, ...].astype(np.float32)
        return inp, W, H, pW, pH, tw, th

    def _to_img_coords(self, poly: np.ndarray, pW: int, pH: int, tw: int, th: int,
                       W: int, H: int) -> np.ndarray:
        """Invert net coords to input-image pixels (subtract margin, clip)."""
        margin = max(0, self.det_margin)
        out = poly.astype(np.float32).copy()
        out[:, 0] = out[:, 0] * (pW / float(tw)) - margin
        out[:, 1] = out[:, 1] * (pH / float(th)) - margin
        out[:, 0] = np.clip(out[:, 0], 0, W)
        out[:, 1] = np.clip(out[:, 1], 0, H)
        return out

    def detect_bgr(self, img_bgr: np.ndarray) -> list[np.ndarray]:
        """Detect lines. Returns 4-point quads in img pixels."""
        return [c["quad"] for c in self.detect_bgr_debug(img_bgr)["kept"]]

    def _forward(self, img_bgr: np.ndarray):
        """Run the DB net. Returns (pred_map, W, H, pW, pH, tw, th)."""
        self.ensure_loaded()
        assert self._sess is not None
        inp, W, H, pW, pH, tw, th = self._preprocess(img_bgr)
        pred_map = self._sess.run(None, {self._input_name: inp})[0][0, 0]
        return pred_map, W, H, pW, pH, tw, th

    def _postprocess(self, pred_map: np.ndarray, W: int, H: int, pW: int, pH: int,
                     tw: int, th: int) -> dict:
        """DB post-processing (thresh -> contours -> box score -> unclip -> quad).

        Returns the ``detect_bgr_debug`` info dict (without params)."""
        info: dict = {"n_contours": 0, "candidates": [], "kept": []}
        if pred_map.size == 0:
            return info
        mask = pred_map > self.thresh
        contours, _ = cv2.findContours(
            (mask.astype(np.uint8)) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        info["n_contours"] = len(contours)
        scored: list[tuple[float, np.ndarray]] = []
        for cnt in contours:
            pts = cnt.squeeze(1)
            if pts.ndim != 2 or pts.shape[0] < 4:
                info["candidates"].append(
                    {"score": None, "kept": False, "reason": "degenerate-contour"}
                )
                continue
            score = box_score_fast(pred_map, pts)
            pre_img = self._to_img_coords(
                pts.astype(np.float32), pW, pH, tw, th, W, H)
            cand: dict = {
                "score": float(score),
                "poly_pre": pre_img,
                "kept": False,
                "reason": "",
            }
            if score < self.box_thresh:
                cand["reason"] = f"box_score {score:.3f} < box_thresh {self.box_thresh}"
                info["candidates"].append(cand)
                continue
            expanded = unclip(pts.astype(np.float32), self.unclip_ratio)
            if expanded is None or len(expanded) == 0:
                cand["reason"] = "unclip failed (empty path)"
                info["candidates"].append(cand)
                continue
            unclipped_img = self._to_img_coords(expanded, pW, pH, tw, th, W, H)
            cand["poly_unclipped"] = unclipped_img
            ux1, uy1 = float(unclipped_img[:, 0].min()), float(unclipped_img[:, 1].min())
            ux2, uy2 = float(unclipped_img[:, 0].max()), float(unclipped_img[:, 1].max())
            if (ux2 - ux1) < self.min_short_side or (uy2 - uy1) < self.min_short_side:
                cand["reason"] = (
                    f"min_short_side: {(ux2 - ux1):.1f}x{(uy2 - uy1):.1f} "
                    f"< {self.min_short_side}"
                )
                info["candidates"].append(cand)
                continue
            # Clean 4-point box with safety padding (Space verbatim).
            (rcx, rcy), (rw, rh), angle = cv2.minAreaRect(expanded.astype(np.float32))
            padded_rect = ((rcx, rcy), (rw + self.box_pad, rh + self.box_pad), angle)
            quad = cv2.boxPoints(padded_rect).astype(np.float32)
            quad_img = self._to_img_coords(quad, pW, pH, tw, th, W, H)
            cand["quad"] = quad_img
            cand["kept"] = True
            info["candidates"].append(cand)
            scored.append((score, quad_img))
        scored.sort(key=lambda t: -t[0])
        if len(scored) > self.max_candidates:
            info["truncated"] = len(scored) - self.max_candidates
            scored = scored[: self.max_candidates]
        info["kept"] = [{"score": float(s), "quad": q, "poly": q} for s, q in scored]
        return info

    def detect_bgr_debug(self, img_bgr: np.ndarray) -> dict:
        """Detect lines with per-candidate diagnostics.

        Returns ``{"params": {...}, "n_contours": int, "candidates": [...],
        "kept": [{"score": float, "quad": (4,2)}]}``. Each candidate records
        the raw contour (``poly_pre``), the unclipped polygon
        (``poly_unclipped``) and the final quad — all in image pixels.
        """
        info: dict = {
            "params": {
                "det_long_side": self.det_long_side,
                "det_min_side": self.det_min_side,
                "det_margin": max(0, self.det_margin),
                "thresh": self.thresh,
                "box_thresh": self.box_thresh,
                "unclip_ratio": self.unclip_ratio,
                "min_short_side": self.min_short_side,
                "box_pad": self.box_pad,
                "max_candidates": self.max_candidates,
            },
        }
        if img_bgr.size == 0:
            info.update({"n_contours": 0, "candidates": [], "kept": []})
            return info
        pred_map, W, H, pW, pH, tw, th = self._forward(img_bgr)
        info.update(self._postprocess(pred_map, W, H, pW, pH, tw, th))
        return info

    def detect_pil(self, img: Image.Image) -> list[np.ndarray]:
        arr = np.array(img.convert("RGB"))[:, :, ::-1]  # RGB -> BGR
        return self.detect_bgr(arr)
