import numpy as np

from comictxt.geometry import (
    clip_box,
    containment_suppress,
    cxcywh_to_xyxy,
    expand_box,
    merge_contained_boxes,
    normalized_bbox_dict,
    nms,
    polygon_to_xyxy,
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
