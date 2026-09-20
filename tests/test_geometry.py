import math

import numpy as np
import pytest

from comictxt.geometry import (
    clip_box,
    containment_suppress,
    cxcywh_to_xyxy,
    expand_box,
    merge_contained_boxes,
    normalized_bbox_dict,
    nms,
    order_quad_corners,
    polygon_to_xyxy,
    warp_quad_to_rect,
    xyxy_to_cxcywh,
)


def test_xyxy_cxcywh_roundtrip():
    box = np.array([10.0, 20.0, 110.0, 60.0])
    cx, cy, w, h = xyxy_to_cxcywh(box)
    assert (cx, cy, w, h) == (60.0, 40.0, 100.0, 40.0)
    assert cxcywh_to_xyxy(cx, cy, w, h) == (10.0, 20.0, 110.0, 60.0)


def test_normalized_bbox_ranges():
    d = normalized_bbox_dict(0, 0, 50, 100, 200, 400)
    assert d["center_x"] == 0.125
    assert d["center_y"] == 0.125
    assert d["width"] == 0.25
    assert d["height"] == 0.25
    assert d["rotation_z"] == 0.0
    for k in ("center_x", "center_y", "width", "height"):
        assert 0.0 <= d[k] <= 1.0


def test_clip_and_expand():
    assert clip_box(-5, -5, 500, 500, 100, 100) == (0, 0, 100, 100)
    x1, y1, x2, y2 = expand_box(40, 40, 60, 60, 1.0, 100, 100)
    assert (x1, y1, x2, y2) == (30, 30, 70, 70)


def test_polygon_to_xyxy():
    poly = np.array([[1, 5], [4, 2], [3, 8]], dtype=np.float32)
    assert polygon_to_xyxy(poly) == (1.0, 2.0, 4.0, 8.0)


def test_nms_suppresses_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    kept = nms(boxes, scores, 0.5)
    assert kept[0] == 0
    assert 1 not in kept
    assert 2 in kept


def test_nms_empty():
    assert nms(np.zeros((0, 4), dtype=np.float32), np.zeros((0,)), 0.5) == []


def test_containment_suppress_drops_nested():
    boxes = np.array(
        [[0, 0, 100, 100], [20, 20, 40, 40], [200, 200, 300, 300]], dtype=np.float32
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    kept = containment_suppress(boxes, scores, 0.9)
    assert kept[0] == 0
    assert 1 not in kept  # fully inside box 0 despite tiny IoU
    assert 2 in kept


def test_containment_suppress_keeps_partial_overlap():
    boxes = np.array([[0, 0, 100, 100], [50, 50, 150, 150]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert containment_suppress(boxes, scores, 0.9) == [0, 1]


def test_containment_suppress_disabled_at_zero():
    boxes = np.array([[0, 0, 100, 100], [20, 20, 40, 40]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    # threshold > 1 can never trigger
    assert containment_suppress(boxes, scores, 1.5) == [0, 1]


def test_merge_contained_inner_higher_conf():
    # the reported bug: inner fragment scores HIGHER than its container,
    # so score-ordered suppression keeps both. merge must collapse them.
    boxes = np.array(
        [[0, 0, 100, 100], [20, 20, 40, 40], [200, 200, 300, 300]], dtype=np.float32
    )
    scores = np.array([0.33, 0.93, 0.5], dtype=np.float32)
    merged, mscores = merge_contained_boxes(boxes, scores, 0.8)
    assert len(merged) == 2
    # union of outer + inner == outer, score is the max
    assert merged[0].tolist() == [0.0, 0.0, 100.0, 100.0]
    assert mscores[0] == 0.93
    assert merged[1].tolist() == [200.0, 200.0, 300.0, 300.0]


def test_merge_contained_chain():
    boxes = np.array(
        [[0, 0, 100, 100], [10, 10, 90, 90], [20, 20, 30, 30]], dtype=np.float32
    )
    scores = np.array([0.2, 0.9, 0.5], dtype=np.float32)
    merged, mscores = merge_contained_boxes(boxes, scores, 0.8)
    assert len(merged) == 1
    assert merged[0].tolist() == [0.0, 0.0, 100.0, 100.0]
    assert mscores[0] == 0.9


def test_merge_contained_keeps_partial_overlap():
    boxes = np.array([[0, 0, 100, 100], [50, 50, 150, 150]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    merged, mscores = merge_contained_boxes(boxes, scores, 0.9)
    assert len(merged) == 2


def test_merge_contained_empty():
    merged, mscores = merge_contained_boxes(
        np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0.8
    )
    assert len(merged) == 0 and len(mscores) == 0


def _rotated_quad(cx=100.0, cy=100.0, w=20.0, h=100.0, deg=20.0):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[cx + dx * c - dy * s, cy + dx * s + dy * c]
         for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2),
                        (w / 2, h / 2), (-w / 2, h / 2))],
        dtype=np.float32,
    )


def test_order_quad_corners():
    quad = _rotated_quad()[::-1]  # shuffled order
    rect = order_quad_corners(quad)
    assert rect.shape == (4, 2)
    # tl has min sum, br max sum
    s = rect.sum(axis=1)
    assert s[0] == s.min() and s[2] == s.max()


def test_warp_quad_to_rect_deskews_rotated():
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    quad = _rotated_quad()
    warped = warp_quad_to_rect(img, quad)
    # deskewed to the true 20x100 line size ...
    assert warped.shape == (100, 20, 3)
    # ... not the inflated ~53x101 axis-aligned bbox
    x1, y1, x2, y2 = polygon_to_xyxy(quad)
    assert (x2 - x1) > 40 and (y2 - y1) > 100
    # vertical stays vertical (no rotate-if-tall)
    assert warped.shape[0] > warped.shape[1]


def test_warp_quad_axis_aligned_is_identity_sized():
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    quad = np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.float32)
    warped = warp_quad_to_rect(img, quad)
    assert warped.shape == (20, 40, 3)


def test_warp_quad_degenerate_falls_back():
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    quad = np.array([[5, 5], [5, 5], [5, 5], [5, 5]], dtype=np.float32)
    warped = warp_quad_to_rect(img, quad)
    assert warped.ndim == 3 and warped.shape[2] == 3


def test_quad_true_size_is_rotation_invariant():
    from comictxt.geometry import quad_true_size

    axis = np.array([[10, 10], [30, 10], [30, 110], [10, 110]], dtype=np.float32)
    w, h = quad_true_size(axis)
    assert (w, h) == (20.0, 100.0)
    rot = _rotated_quad(cx=100.0, cy=100.0, w=20.0, h=100.0, deg=30.0)
    w, h = quad_true_size(rot)
    assert w == pytest.approx(20.0, abs=1.0)
    assert h == pytest.approx(100.0, abs=1.0)


def test_ink_size_measures_ink_not_box():
    from comictxt.geometry import ink_size

    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    # fat black stripe down the middle of a 40px-wide area
    img[10:90, 44:56, :] = 0
    quad = np.array([[40, 5], [60, 5], [60, 95], [40, 95]], dtype=np.float32)
    w, h = ink_size(img, quad)
    assert w == pytest.approx(12, abs=1)
    assert h == pytest.approx(80, abs=1)


def test_ink_size_blank_returns_none():
    from comictxt.geometry import ink_size

    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    quad = np.array([[40, 5], [60, 5], [60, 95], [40, 95]], dtype=np.float32)
    assert ink_size(img, quad) is None


def test_expand_box_modes():
    # proportional: each axis by its own length (w 20->30, h 100->150)
    assert expand_box(40, 0, 60, 100, 0.5, 1000, 1000, mode="proportional") == (35, 0, 65, 125)
    # uniform: smallest side grows by the same absolute amount as the biggest
    # (w=20,h=100 -> pad=0.5*100=50 -> 70 x 150)
    assert expand_box(40, 0, 60, 100, 0.5, 1000, 1000, mode="uniform") == (15, 0, 85, 125)
    # max: pad = ratio*max on every side (w 20->120, h 100->200)
    assert expand_box(40, 0, 60, 100, 0.5, 1000, 1000, mode="max") == (0, 0, 110, 150)


def test_expand_region_boxes_auto_fallback():
    from comictxt.geometry import expand_region_boxes

    isolated_long = (0, 0, 20, 100)
    # no neighbors: auto -> uniform
    out = expand_region_boxes([isolated_long], 0.5, "auto", 1000, 1000)
    assert out == [expand_box(*isolated_long, 0.5, 1000, 1000, mode="uniform")]
    # neighbor nearby: auto -> proportional (would swallow the neighbor)
    near = (30, 0, 50, 100)
    out = expand_region_boxes([isolated_long, near], 0.5, "auto", 1000, 1000)
    assert out[0] == expand_box(*isolated_long, 0.5, 1000, 1000, mode="proportional")
    # not elongated: auto -> proportional even when isolated
    square = (0, 0, 100, 100)
    out = expand_region_boxes([square], 0.5, "auto", 1000, 1000)
    assert out == [expand_box(*square, 0.5, 1000, 1000, mode="proportional")]
