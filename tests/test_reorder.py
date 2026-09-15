"""Reading-order tests for the Space column/row sort port (on by default)."""
from comictxt.config import ComictxtConfig
from comictxt.grouping import (
    apply_reading_order,
    reorder_lines_in_paragraph,
    reorder_paragraphs,
    sort_reading_order,
    vote_vertical_boxes,
)


def _para(cx, cy, w=0.1, h=0.1, direction="LEFT_TO_RIGHT", texts=("a",)):
    return {
        "bounding_box": {"center_x": cx, "center_y": cy, "width": w, "height": h,
                         "rotation_z": 0.0},
        "lines": [
            {"text": t,
             "bounding_box": {"center_x": cx, "center_y": cy, "width": w / 2,
                              "height": h / 2, "rotation_z": 0.0},
             "words": [{"text": t, "bounding_box": {}}]}
            for t in texts
        ],
        "writing_direction": direction,
    }


def _box(cx, cy, w, h):
    return {"xmin": cx - w / 2, "xmax": cx + w / 2,
            "ymin": cy - h / 2, "ymax": cy + h / 2, "cx": cx, "cy": cy}


def test_reorder_on_by_default():
    cfg = ComictxtConfig()
    assert cfg.pipeline.reorder_blocks is True
    assert cfg.pipeline.reorder_lines is True


def test_vote_vertical_boxes():
    assert vote_vertical_boxes([_box(0.5, 0.5, 0.1, 0.4)]) is True
    assert vote_vertical_boxes([_box(0.5, 0.5, 0.4, 0.1)]) is False


def test_vertical_columns_read_right_to_left():
    # Two side-by-side columns: right column first, top-to-bottom within each.
    items = [
        ("left-top", _box(0.3, 0.2, 0.1, 0.2)),
        ("left-bottom", _box(0.3, 0.6, 0.1, 0.2)),
        ("right-top", _box(0.7, 0.25, 0.1, 0.2)),
        ("right-bottom", _box(0.7, 0.65, 0.1, 0.2)),
    ]
    out = sort_reading_order(
        [t for t, _ in items],
        box_of=lambda t: dict(items)[t],
        is_vertical=True,
    )
    assert out == ["right-top", "right-bottom", "left-top", "left-bottom"]


def test_horizontal_rows_read_top_to_bottom():
    items = [
        ("bottom", _box(0.5, 0.7, 0.6, 0.1)),
        ("top", _box(0.5, 0.2, 0.6, 0.1)),
    ]
    out = sort_reading_order(
        [t for t, _ in items],
        box_of=lambda t: dict(items)[t],
        is_vertical=False,
    )
    assert out == ["top", "bottom"]


def test_vertical_lines_sorted_right_to_left():
    para = _para(0.5, 0.5, direction="TOP_TO_BOTTOM",
                 texts=("left", "right"))
    para["lines"][0]["bounding_box"]["center_x"] = 0.3
    para["lines"][1]["bounding_box"]["center_x"] = 0.7
    reorder_lines_in_paragraph(para)
    assert [ln["text"] for ln in para["lines"]] == ["right", "left"]


def test_horizontal_lines_sorted_top_to_bottom():
    para = _para(0.5, 0.5, direction="LEFT_TO_RIGHT",
                 texts=("bottom", "top"))
    para["lines"][0]["bounding_box"]["center_y"] = 0.7
    para["lines"][1]["bounding_box"]["center_y"] = 0.3
    reorder_lines_in_paragraph(para)
    assert [ln["text"] for ln in para["lines"]] == ["top", "bottom"]


def test_paragraph_columns_right_to_left():
    # Tall (vertical-vote) boxes side by side -> right column first.
    left = _para(0.3, 0.5, w=0.1, h=0.5)
    right = _para(0.7, 0.5, w=0.1, h=0.5)
    assert reorder_paragraphs([left, right]) == [right, left]


def test_paragraph_rows_top_to_bottom():
    # Wide (horizontal-vote) boxes stacked -> top row first.
    top = _para(0.5, 0.2, w=0.6, h=0.1)
    bottom = _para(0.5, 0.8, w=0.6, h=0.1)
    assert reorder_paragraphs([bottom, top]) == [top, bottom]


def test_apply_reading_order_flags_off():
    v = _para(0.3, 0.5, direction="TOP_TO_BOTTOM")
    h = _para(0.7, 0.5, direction="LEFT_TO_RIGHT")
    assert apply_reading_order([v, h], reorder_blocks=False,
                               reorder_lines=False) == [v, h]


def test_single_paragraph_untouched():
    p = _para(0.5, 0.5)
    assert reorder_paragraphs([p]) == [p]


def test_tilted_lines_use_deskewed_size_for_columns():
    """Ground-truth 032 region0: three ~30deg-tilted lines whose inflated
    axis boxes collapse into one column (wrong ymin order); the deskewed
    _sort_wh sizes must split them into right-to-left columns."""
    para = _para(0.5, 0.5, direction="TOP_TO_BOTTOM",
                 texts=("mid", "left", "right"))
    axis = [
        (0.131, 0.357, 0.083, 0.079),  # mid
        (0.109, 0.347, 0.078, 0.076),  # left
        (0.167, 0.351, 0.044, 0.033),  # right
    ]
    for ln, (cx, cy, w, h) in zip(para["lines"], axis):
        ln["bounding_box"].update(center_x=cx, center_y=cy, width=w, height=h)
        ln["_sort_wh"] = [0.022, 0.077]
    reorder_lines_in_paragraph(para)
    assert [ln["text"] for ln in para["lines"]] == ["right", "mid", "left"]


def test_tilted_lines_axis_boxes_alone_merge_columns():
    """Sanity: without _sort_wh the same boxes collapse (documents why the
    deskewed size is needed)."""
    para = _para(0.5, 0.5, direction="TOP_TO_BOTTOM",
                 texts=("mid", "left", "right"))
    axis = [
        (0.131, 0.357, 0.083, 0.079),
        (0.109, 0.347, 0.078, 0.076),
        (0.167, 0.351, 0.044, 0.033),
    ]
    for ln, (cx, cy, w, h) in zip(para["lines"], axis):
        ln["bounding_box"].update(center_x=cx, center_y=cy, width=w, height=h)
    reorder_lines_in_paragraph(para)
    assert [ln["text"] for ln in para["lines"]] != ["right", "mid", "left"]


def test_lines_without_sort_wh_fall_back_to_axis_boxes():
    para = _para(0.5, 0.5, direction="TOP_TO_BOTTOM",
                 texts=("left", "right"))
    para["lines"][0]["bounding_box"]["center_x"] = 0.3
    para["lines"][1]["bounding_box"]["center_x"] = 0.7
    reorder_lines_in_paragraph(para)
    assert [ln["text"] for ln in para["lines"]] == ["right", "left"]
