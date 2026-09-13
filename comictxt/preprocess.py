"""Optional image cleanup: levels stretch + unsharp mask.

Levels: pixels <= black_point map to 0, >= white_point map to 255, linear
in between (per channel). Sharpen: unsharp mask with the given amount.
Both are no-ops at default settings (0/255/0.0).
"""
from __future__ import annotations

import cv2
import numpy as np


def build_levels_lut(black_point: int = 0, white_point: int = 255) -> np.ndarray:
    """256-entry LUT for the levels stretch."""
    black = max(0, min(255, int(black_point)))
    white = max(0, min(255, int(white_point)))
    if white <= black:
        white = min(255, black + 1)
    lut = np.arange(256, dtype=np.float32)
    lut = (lut - black) * (255.0 / (white - black))
    return np.clip(lut, 0, 255).astype(np.uint8)


def apply_levels(img_bgr: np.ndarray, black_point: int = 0, white_point: int = 255) -> np.ndarray:
    """Apply levels stretch per channel; identity at 0/255."""
    if int(black_point) <= 0 and int(white_point) >= 255:
        return img_bgr
    lut = build_levels_lut(black_point, white_point)
    return lut[img_bgr]


def apply_sharpen(img_bgr: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """Unsharp mask; identity at amount <= 0."""
    if float(amount) <= 0:
        return img_bgr
    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(
        img_bgr.astype(np.float32), 1.0 + float(amount),
        blur.astype(np.float32), -float(amount), 0,
    )
    return np.clip(sharp, 0, 255).astype(np.uint8)


def cleanup(img_bgr: np.ndarray, enable: bool = False, black_point: int = 0,
            white_point: int = 255, sharpen: float = 0.0) -> np.ndarray:
    """Levels then sharpen; passthrough when disabled."""
    if not enable:
        return img_bgr
    out = apply_levels(img_bgr, black_point, white_point)
    return apply_sharpen(out, sharpen)
