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
    x1: float, y1: float, x2: float, y2: float, ratio: float, W: int, H: int
) -> tuple[int, int, int, int]:
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nw, nh = w * (1.0 + ratio), h * (1.0 + ratio)
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


def polygon_to_xyxy(poly: np.ndarray) -> tuple[float, float, float, float]:
    xs = poly[:, 0]
    ys = poly[:, 1]
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


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
