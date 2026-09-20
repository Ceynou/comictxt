"""Geometry helpers: box conversions, clipping, expansion, NMS."""
from __future__ import annotations

import math

import numpy as np


def xyxy_to_cxcywh(box: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)


def cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def clip_box(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> tuple[int, int, int, int]:
    x1 = int(round(max(0, min(x1, W))))
    y1 = int(round(max(0, min(y1, H))))
    x2 = int(round(max(0, min(x2, W))))
    y2 = int(round(max(0, min(y2, H))))
    return (x1, y1, x2, y2)


def expand_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    ratio: float,
    W: int,
    H: int,
    mode: str = "uniform",
) -> tuple[int, int, int, int]:
    """Expand a box by ``ratio``.

    ``mode="proportional"`` grows each axis by its own length
    (w*(1+r), h*(1+r)): tight long-line crops stay tight across their
    narrow side — the length soaks up all the padding.

    ``mode="uniform"`` (default) expands the smallest side by the same
    absolute amount as the biggest side (pad = ratio * max(w, h) on both
    axes), so a tall narrow line region gains real context left/right
    instead of only along its length.

    ``mode="max"`` adds pad = ratio * max(w, h) to every side (the
    biggest-side rate on all four sides).
    """
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if mode == "proportional":
        nw, nh = w * (1.0 + ratio), h * (1.0 + ratio)
    elif mode == "max":
        pad = ratio * max(w, h)
        nw, nh = w + 2.0 * pad, h + 2.0 * pad
    else:  # uniform
        pad = ratio * max(w, h)
        nw, nh = w + pad, h + pad
    return clip_box(cx - nw / 2.0, cy - nh / 2.0, cx + nw / 2.0, cy + nh / 2.0, W, H)


def normalized_bbox_dict(
    x1: float, y1: float, x2: float, y2: float, W: int, H: int, rotation_z: float = 0.0
) -> dict:
    cx, cy, w, h = xyxy_to_cxcywh(np.array([x1, y1, x2, y2], dtype=np.float64))
    return {
        "center_x": cx / max(1, W),
        "center_y": cy / max(1, H),
        "width": w / max(1, W),
        "height": h / max(1, H),
        "rotation_z": float(rotation_z),
    }


def box_overlap(a, b) -> float:
    """Overlap area of two xyxy boxes (0.0 when disjoint)."""
    ix1, iy1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix2, iy2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def expand_region_boxes(
    boxes: list, ratio: float, mode: str, W: int, H: int, min_aspect: float = 2.0
) -> list[tuple]:
    """Expand a list of region boxes per ``region.pad_mode``.

    ``auto`` (default): strongly elongated, isolated regions (the tight
    long-line case) expand ``uniform`` — the smallest side grows by the
    same absolute amount as the biggest side, so the crop gains real
    context left/right instead of only along its length. Anything less
    elongated than ``min_aspect`` (or whose uniform box would reach into
    another detected region) falls back to ``proportional``, so adjacent
    bubbles/columns are not swallowed into each other's crops.
    """
    out: list[tuple] = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = (float(v) for v in b)
        w, h = x2 - x1, y2 - y1
        m = mode
        if mode == "auto":
            elongated = max(w, h) >= min_aspect * max(1.0, min(w, h))
            wide = expand_box(x1, y1, x2, y2, ratio, W, H, mode="uniform")
            others = [boxes[j] for j in range(len(boxes)) if j != i]
            blocked = any(box_overlap(wide, o) > 0.0 for o in others)
            m = "uniform" if elongated and not blocked else "proportional"
        out.append(expand_box(x1, y1, x2, y2, ratio, W, H, mode=m))
    return out


def polygon_to_xyxy(poly: np.ndarray) -> tuple[float, float, float, float]:
    xs = poly[:, 0]
    ys = poly[:, 1]
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def order_quad_corners(quad: np.ndarray) -> np.ndarray:
    """Order 4 points as (tl, tr, br, bl) via sum/diff heuristic.

    Same convention as the PP-OCR Space app's ``get_rotate_crop_image``.
    """
    pts = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def quad_true_size(quad: np.ndarray) -> tuple[float, float]:
    """Deskewed (w, h) of a detector quad from its edge lengths.

    Axis-aligned ``polygon_to_xyxy`` boxes inflate tilted lines (a 21px-wide
    line at 30deg measures ~53px wide), which breaks width-based thresholds
    in reading-order sorting and furigana geometry. Edge lengths are
    rotation-invariant: w = mean(top, bottom), h = mean(left, right).
    """
    rect = order_quad_corners(np.asarray(quad, dtype=np.float32).reshape(-1, 2))
    tl, tr, br, bl = rect
    top = float(np.linalg.norm(tr - tl))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))
    right = float(np.linalg.norm(br - tr))
    return ((top + bottom) / 2.0, (left + right) / 2.0)


def ink_size(img_bgr: np.ndarray, quad: np.ndarray) -> tuple[float, float] | None:
    """Ink (w, h) inside a detector quad, via warp + Otsu bbox.

    Detector quads carry ``box_pad`` inflation and minAreaRect slack, so
    box thickness ratios are noisy for furigana decisions; the actual ink
    extent is stable (ruby ink is reliably ~50-60% of the main line's,
    while det boxes wander up to ~0.85). Returns None when no ink is found.
    """
    import cv2

    warped = warp_quad_to_rect(img_bgr, np.asarray(quad, dtype=np.float32))
    if warped is None or warped.size == 0 or warped.shape[0] < 2 or warped.shape[1] < 2:
        return None
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rows = np.where(binv.sum(axis=1) > 0)[0]
    cols = np.where(binv.sum(axis=0) > 0)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return (float(cols[-1] - cols[0] + 1), float(rows[-1] - rows[0] + 1))


def warp_quad_to_rect(
    img_bgr: np.ndarray, quad: np.ndarray, border_value: tuple[int, int, int] | None = None
) -> np.ndarray:
    """Perspective-warp a detector quad to a tight upright rectangle.

    Generic (recognizer-agnostic) deskew for rotated ``minAreaRect`` quads:
    unlike ``rec_ppocr.warp_quad`` there is no Otsu trim and no
    rotate-if-tall — vertical crops stay vertical for Hayai's 2D-RoPE.
    Falls back to an axis-aligned crop when the quad is degenerate.
    """
    import cv2

    pts = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4 or img_bgr.size == 0:
        x1, y1, x2, y2 = polygon_to_xyxy(pts.reshape(-1, 2)) if pts.size else (0, 0, 0, 0)
        H, W = img_bgr.shape[:2]
        ix1, iy1, ix2, iy2 = clip_box(x1, y1, x2, y2, W, H)
        if ix2 <= ix1 or iy2 <= iy1:
            return np.zeros((8, 8, 3), dtype=np.uint8)
        return img_bgr[iy1:iy2, ix1:ix2].copy()
    rect = order_quad_corners(pts)
    (tl, tr, br, bl) = rect
    width_a = float(np.linalg.norm(br - bl))
    width_b = float(np.linalg.norm(tr - tl))
    height_a = float(np.linalg.norm(tr - br))
    height_b = float(np.linalg.norm(tl - bl))
    max_width = max(int(round(width_a)), int(round(width_b)))
    max_height = max(int(round(height_a)), int(round(height_b)))
    if max_width <= 0 or max_height <= 0:
        return np.zeros((8, 8, 3), dtype=np.uint8)
    # Keep output bounded: detector quads are line-sized; cap at 1024px
    # to avoid pathological memory use on bad detections.
    max_width = min(max_width, 1024)
    max_height = min(max_height, 1024)
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    kwargs: dict = {"borderMode": cv2.BORDER_REPLICATE, "flags": cv2.INTER_LINEAR}
    if border_value is not None:
        kwargs = {"borderMode": cv2.BORDER_CONSTANT, "borderValue": border_value,
                  "flags": cv2.INTER_LINEAR}
    return cv2.warpPerspective(img_bgr, M, (max_width, max_height), **kwargs)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Greedy NMS on xyxy boxes. Returns kept indices."""
    if len(boxes) == 0:
        return []
    order = np.argsort(-scores)
    keep: list[int] = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        for j in order:
            if j == i or suppressed[j]:
                continue
            if box_iou(boxes[i], boxes[j]) > iou_thresh:
                suppressed[j] = True
    return keep


def containment_suppress(
    boxes: np.ndarray, scores: np.ndarray, contain_thresh: float = 0.9
) -> list[int]:
    """Drop boxes mostly contained inside a higher-scored box.

    NMS (IoU-based) never fires when the inner box is much smaller than the
    outer one. Returns kept indices, highest score first.
    """
    if len(boxes) == 0:
        return []
    order = list(np.argsort(-scores))
    keep: list[int] = []
    for i in order:
        contained = False
        for k in keep:
            ix1 = max(boxes[i][0], boxes[k][0])
            iy1 = max(boxes[i][1], boxes[k][1])
            ix2 = min(boxes[i][2], boxes[k][2])
            iy2 = min(boxes[i][3], boxes[k][3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_i = max(0.0, boxes[i][2] - boxes[i][0]) * max(0.0, boxes[i][3] - boxes[i][1])
            if area_i > 0 and inter / area_i >= contain_thresh:
                contained = True
                break
        if not contained:
            keep.append(int(i))
    return keep


def _containment_ratio(inner: np.ndarray, outer: np.ndarray) -> float:
    """Fraction of inner's area covered by outer."""
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return inter / area if area > 0 else 0.0


def merge_contained_boxes(
    boxes: np.ndarray, scores: np.ndarray, contain_thresh: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
    """Merge boxes mostly contained inside another box (union + max score).

    Unlike score-ordered suppression, this handles the common YOLO case where
    the inner fragment scores HIGHER than its container: clustering is
    biggest-area-first, so the encompassing box always absorbs its fragments
    regardless of score order. NMS (IoU-based) never fires here because the
    inner box is much smaller than the outer one.
    """
    if len(boxes) == 0:
        return boxes, scores
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = list(np.argsort(-areas))
    clusters: list[int] = []  # kept-box index per cluster in first-seen order
    kept_boxes: list[np.ndarray] = []
    kept_scores: list[float] = []
    for i in order:
        placed = False
        for c, kb in enumerate(kept_boxes):
            if _containment_ratio(boxes[i], kb) >= contain_thresh:
                kb[0] = min(kb[0], boxes[i][0])
                kb[1] = min(kb[1], boxes[i][1])
                kb[2] = max(kb[2], boxes[i][2])
                kb[3] = max(kb[3], boxes[i][3])
                kept_scores[c] = max(kept_scores[c], float(scores[i]))
                placed = True
                break
        if not placed:
            clusters.append(int(i))
            kept_boxes.append(boxes[i].astype(np.float64).copy())
            kept_scores.append(float(scores[i]))
    if not kept_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return (
        np.stack(kept_boxes, axis=0).astype(np.float32),
        np.array(kept_scores, dtype=np.float32),
    )


def merge_boxes_to_xyxy(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2)


def line_corners_from_cxcywhr(
    cx: float, cy: float, w: float, h: float, rot: float
) -> list[list[float]]:
    cos_a = math.cos(rot)
    sin_a = math.sin(rot)
    corners = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        corners.append([round(cx + rx, 1), round(cy + ry, 1)])
    return corners
