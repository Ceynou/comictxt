"""PP-OCRv6 manga text-line recognizer (CTC, ONNX).

Port of the recognition half of Kellenok's PP-OCR_manga Space app
(``Kellenok/PP-OCRv6_manga`` Spaces ``app.py``): perspective warp of the
detector quad (``get_rotate_crop_image``), Otsu-projection furigana/margin
trim (``clean_manga_vertical_crop``), rotate-if-tall, resize to height 48,
``(x-0.5)/0.5`` normalize, greedy CTC decode with the Space's vocab.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image


def _hf_hub_roots() -> list[Path]:
    from comictxt.config import _hf_hub_roots as _roots

    return _roots()


def resolve_ppocr_model(explicit: str = "", use_fp16: bool = False) -> Path:
    """Resolve ``rec/manga_rec_v0.1[_fp16].onnx`` from the local HF cache."""
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"ppocr rec model not found: {p}")
        return p
    names = (
        ["manga_rec_v0.1_fp16.onnx", "manga_rec_v0.1.onnx"]
        if use_fp16
        else ["manga_rec_v0.1.onnx", "manga_rec_v0.1_fp16.onnx"]
    )
    for root in _hf_hub_roots():
        snaps = root / "models--Kellenok--PP-OCRv6_manga" / "snapshots"
        if not snaps.is_dir():
            continue
        for sha_dir in sorted(snaps.iterdir()):
            for name in names:
                cand = sha_dir / "rec" / name
                if cand.is_file():
                    # Prefer the requested precision; fall back otherwise.
                    if name == names[0] or not (sha_dir / "rec" / names[0]).is_file():
                        return cand
    raise FileNotFoundError(
        "Could not auto-resolve PP-OCR rec model in HF cache. "
        "Set rec.ppocr_model explicitly."
    )


def resolve_ppocr_dict(explicit: str = "") -> Path:
    """Resolve ``ppocrv6_dict.txt`` from the local HF cache."""
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"ppocr dict not found: {p}")
        return p
    for root in _hf_hub_roots():
        snaps = root / "models--Kellenok--PP-OCRv6_manga" / "snapshots"
        if not snaps.is_dir():
            continue
        for sha_dir in sorted(snaps.iterdir()):
            cand = sha_dir / "ppocrv6_dict.txt"
            if cand.is_file():
                return cand
    raise FileNotFoundError(
        "Could not auto-resolve ppocrv6_dict.txt in HF cache. "
        "Set rec.ppocr_dict explicitly."
    )


def _trim_sides_vertical(c: np.ndarray) -> np.ndarray:
    """Right (furigana) + left (margin) trim on a vertical-frame crop.

    Shared core of the Space's ``get_rotate_crop_image`` trim block and
    ``clean_manga_vertical_crop``. Input must be taller than wide; output is
    the trimmed crop in the same orientation.
    """
    h, w = c.shape[:2]
    if h <= w or (w / float(h)) < 0.25 or w < 24:
        return c
    gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if c.ndim == 3 else c
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = np.sum(bin_img, axis=0)
    x1, x2 = 0, w

    # 1. Right trim (furigana alongside the main column)
    r_start, r_end = int(w * 0.55), int(w * 0.90)
    zero_cols = [x for x in range(r_start, r_end) if proj[x] == 0]
    if zero_cols:
        split_r = zero_cols[0]
        right_mask = bin_img[:, split_r:]
        right_rows = np.where(np.sum(right_mask, axis=1) > 0)[0]
        if len(right_rows) >= 8:
            left_mask = bin_img[:, :split_r]
            left_rows = np.sum(left_mask, axis=1) > 0
            overlap = np.sum(left_rows[right_rows]) if len(right_rows) else 0
            alongside_ratio = overlap / float(len(right_rows)) if len(right_rows) else 0
            right_area = np.sum(right_mask > 0)
            left_area = np.sum(left_mask > 0)
            if (alongside_ratio >= 0.85 and 15 <= right_area <= 0.35 * left_area
                    and right_rows[0] < int(h * 0.75)):
                x2 = split_r

    # 2. Symmetric left trim (stray margin / bubble border noise)
    l_start = max(3, int(w * 0.10))
    l_end = int(w * 0.25)
    l_zeros = [x for x in range(l_start, l_end) if proj[x] == 0]
    if l_zeros:
        split_l = l_zeros[-1] + 1
        left_mask = bin_img[:, :split_l]
        left_rows = np.where(np.sum(left_mask, axis=1) > 0)[0]
        if len(left_rows) >= 8:
            mid_mask = bin_img[:, split_l:x2]
            mid_rows = np.sum(mid_mask, axis=1) > 0
            overlap = np.sum(mid_rows[left_rows]) if len(left_rows) else 0
            alongside_ratio = overlap / float(len(left_rows)) if len(left_rows) else 0
            left_area = np.sum(left_mask > 0)
            mid_area = np.sum(mid_mask > 0)
            if (alongside_ratio >= 0.85 and 15 <= left_area <= 0.35 * mid_area
                    and left_rows[0] < int(h * 0.75)):
                x1 = split_l

    if (x2 - x1) >= 16:
        return c[:, x1:x2]
    return c


def ctc_greedy_decode(indices, vocab: list[str]) -> str:
    """Greedy CTC decode: drop blanks (0), collapse repeats (Space verbatim)."""
    return "".join(
        vocab[idx]
        for i, idx in enumerate(indices)
        if idx != 0 and (i == 0 or idx != indices[i - 1])
    )


def trim_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Space ``clean_manga_vertical_crop``: trim in vertical frame, restore."""
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 4 or crop_bgr.shape[1] < 4:
        return crop_bgr
    was_horizontal = crop_bgr.shape[0] < crop_bgr.shape[1]
    c = cv2.rotate(crop_bgr, cv2.ROTATE_90_CLOCKWISE) if was_horizontal else crop_bgr
    c_clean = _trim_sides_vertical(c)
    if c_clean is not c and (c_clean.shape[1] >= 16):
        return cv2.rotate(c_clean, cv2.ROTATE_90_COUNTERCLOCKWISE) if was_horizontal else c_clean
    return crop_bgr


def warp_quad(img_bgr: np.ndarray, quad: np.ndarray, trim: bool = True):
    """Space ``get_rotate_crop_image``: perspective warp + trim + rotate.

    Returns ``(warped_bgr, new_quad)`` with ``new_quad`` in input-image
    coordinates (updated through the trim, like the Space).
    """
    points = np.asarray(quad, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    widthA = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
    widthB = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    heightB = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    maxHeight = max(int(heightA), int(heightB))
    if maxWidth <= 0 or maxHeight <= 0:
        return np.zeros((10, 10, 3), dtype=np.uint8), points
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        img_bgr, M, (maxWidth, maxHeight),
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LINEAR,
    )
    new_points = points

    if trim:
        # Faithful block: operate exactly like the Space on `warped`.
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        proj = np.sum(bin_img, axis=0)
        x1, x2 = 0, maxWidth
        if maxHeight > maxWidth and (maxWidth / float(maxHeight)) >= 0.25 and maxWidth >= 24:
            r_start, r_end = int(maxWidth * 0.55), int(maxWidth * 0.90)
            zero_cols = [x for x in range(r_start, r_end) if proj[x] == 0]
            if zero_cols:
                split_r = zero_cols[0]
                right_mask = bin_img[:, split_r:]
                right_rows = np.where(np.sum(right_mask, axis=1) > 0)[0]
                if len(right_rows) >= 8:
                    left_mask = bin_img[:, :split_r]
                    left_rows = np.sum(left_mask, axis=1) > 0
                    overlap = np.sum(left_rows[right_rows]) if len(right_rows) else 0
                    alongside_ratio = overlap / float(len(right_rows)) if len(right_rows) else 0
                    right_area = np.sum(right_mask > 0)
                    left_area = np.sum(left_mask > 0)
                    if (alongside_ratio >= 0.85 and 15 <= right_area <= 0.35 * left_area
                            and right_rows[0] < int(maxHeight * 0.75)):
                        x2 = split_r
            l_start = max(3, int(maxWidth * 0.10))
            l_end = int(maxWidth * 0.25)
            l_zeros = [x for x in range(l_start, l_end) if proj[x] == 0]
            if l_zeros:
                split_l = l_zeros[-1] + 1
                left_mask = bin_img[:, :split_l]
                left_rows = np.where(np.sum(left_mask, axis=1) > 0)[0]
                if len(left_rows) >= 8:
                    mid_mask = bin_img[:, split_l:x2]
                    mid_rows = np.sum(mid_mask, axis=1) > 0
                    overlap = np.sum(mid_rows[left_rows]) if len(left_rows) else 0
                    alongside_ratio = overlap / float(len(left_rows)) if len(left_rows) else 0
                    left_area = np.sum(left_mask > 0)
                    mid_area = np.sum(mid_mask > 0)
                    if (alongside_ratio >= 0.85 and 15 <= left_area <= 0.35 * mid_area
                            and left_rows[0] < int(maxHeight * 0.75)):
                        x1 = split_l
            if (x2 - x1) >= 16 and (x1 > 0 or x2 < maxWidth):
                warped = warped[:, x1:x2]
                _, M_inv = cv2.invert(M)
                trimmed_corners = np.array(
                    [[[x1, 0], [x2 - 1, 0], [x2 - 1, maxHeight - 1], [x1, maxHeight - 1]]],
                    dtype=np.float32,
                )
                new_points = cv2.perspectiveTransform(trimmed_corners, M_inv)[0]

    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return warped, np.asarray(new_points, dtype=np.float32)


class PpocrRecognizer:
    """PP-OCRv6 manga CTC recognizer (ONNX). Thread-safe via a lock."""

    def __init__(
        self,
        model_path: Union[str, Path],
        dict_path: Union[str, Path],
        providers: Optional[list[str]] = None,
        trim: bool = True,
        preprocess: Optional[dict] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.dict_path = Path(dict_path)
        self.providers = providers or ["CPUExecutionProvider"]
        self.trim = bool(trim)
        self.preprocess = dict(preprocess) if preprocess else {"enable": False}
        self._sess = None
        self._input_name = "x"
        self._vocab: list[str] = []
        self._lock = threading.Lock()

    def load(self) -> "PpocrRecognizer":
        import onnxruntime as ort

        if not self.model_path.is_file():
            raise FileNotFoundError(f"PP-OCR rec model not found: {self.model_path}")
        if not self.dict_path.is_file():
            raise FileNotFoundError(f"PP-OCR dict not found: {self.dict_path}")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=self.providers
        )
        try:
            self._input_name = self._sess.get_inputs()[0].name
        except Exception:
            self._input_name = "x"
        with open(self.dict_path, encoding="utf-8") as f:
            self._vocab = ["blank"] + [line.strip("\r\n") for line in f] + [" "]
        return self

    def ensure_loaded(self) -> None:
        if self._sess is None:
            self.load()

    # -- core -------------------------------------------------------------
    def _decode_bgr(self, crop_bgr: np.ndarray) -> str:
        """Trim (optional) + rotate-if-tall + resize-h48 + CTC decode."""
        self.ensure_loaded()
        assert self._sess is not None
        if crop_bgr.size == 0 or crop_bgr.shape[0] < 2 or crop_bgr.shape[1] < 2:
            return ""
        crop = trim_crop(crop_bgr) if self.trim else crop_bgr
        if crop.shape[0] > crop.shape[1]:
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        pp = self.preprocess or {}
        if pp.get("enable"):
            from comictxt.preprocess import cleanup as cleanup_image

            crop = cleanup_image(
                np.ascontiguousarray(crop),
                enable=True,
                black_point=int(pp.get("black_point", 0)),
                white_point=int(pp.get("white_point", 255)),
                sharpen=float(pp.get("sharpen", 0.0)),
            )
        target_w = max(16, min(640, int(round(48.0 * crop.shape[1] / max(1, crop.shape[0])))))
        c_inp = cv2.resize(crop, (target_w, 48)).astype(np.float32) / 255.0
        c_inp = ((c_inp - 0.5) / 0.5).transpose((2, 0, 1))[np.newaxis, ...]
        with self._lock:
            logits = self._sess.run(None, {self._input_name: c_inp})[0][0]
        indices = np.argmax(logits, axis=-1)
        return ctc_greedy_decode(indices, self._vocab)

    def ocr_quad(self, img_bgr: np.ndarray, quad: np.ndarray) -> tuple[str, np.ndarray]:
        """Warp a detector quad, recognize it. Returns (text, updated quad)."""
        warped, new_quad = warp_quad(img_bgr, quad, trim=self.trim)
        return self._decode_bgr(warped), new_quad

    def ocr_pil(self, img: Image.Image) -> str:
        arr = np.array(img.convert("RGB"))[:, :, ::-1]  # RGB -> BGR
        return self._decode_bgr(arr)

    def ocr_array(self, arr: np.ndarray) -> str:
        if arr.ndim == 2:
            img = np.stack([arr] * 3, axis=-1)
        elif arr.shape[2] == 4:
            img = arr[:, :, :3]
        else:
            img = arr
        return self._decode_bgr(np.ascontiguousarray(img[:, :, ::-1]))


def _resolve_env_offline() -> bool:
    return any(
        os.environ.get(v, "").strip().lower() in ("1", "true", "yes")
        for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "COMICTXT_OFFLINE")
    )
