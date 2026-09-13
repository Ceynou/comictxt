from comictxt.grouping import build_paragraph, infer_orientation


def test_build_paragraph_horizontal():
    para = build_paragraph(
        [
            {"text": "hello", "xyxy": (0, 0, 100, 20)},
            {"text": "world", "xyxy": (0, 30, 100, 50)},
        ],
        200,
        200,
    )
    assert para is not None
    assert para["writing_direction"] == "LEFT_TO_RIGHT"
    assert len(para["lines"]) == 2
    assert para["lines"][0]["text"] == "hello"
    bb = para["lines"][0]["bounding_box"]
    assert 0.0 <= bb["center_x"] <= 1.0
    assert para["bounding_box"]["width"] > 0


def test_side_by_side_lines_are_vertical():
    # ground_truth/131.json block [985,355,1113,634]: 3 columns, y-overlapping
    boxes = [
        (1066, 356, 1107, 531),
        (1029, 358, 1066, 634),
        (985, 359, 1023, 567),
    ]
    assert infer_orientation(boxes) is True
    para = build_paragraph(
        [{"text": t, "xyxy": b} for t, b in zip(["a", "b", "c"], boxes)],
        1351,
        1920,
    )
    assert para is not None
    assert para["writing_direction"] == "TOP_TO_BOTTOM"


def test_stacked_lines_are_horizontal():
    # ground_truth/131.json block [11,64,965,228]: 2 rows stacked along y
    boxes = [(24, 64, 565, 140), (11, 150, 965, 228)]
    assert infer_orientation(boxes) is False
    para = build_paragraph(
        [{"text": t, "xyxy": b} for t, b in zip(["a", "b"], boxes)],
        1351,
        1920,
    )
    assert para is not None
    assert para["writing_direction"] == "LEFT_TO_RIGHT"


def test_single_line_uses_region_aspect_not_line_aspect():
    # short wide line inside a tall bubble region -> vertical (region wins)
    para = build_paragraph(
        [{"text": "…仕事？", "xyxy": (1030, 1200, 1087, 1240)}],
        1351,
        1920,
        region_xyxy=(1030, 1123, 1087, 1320),
    )
    assert para is not None
    assert para["writing_direction"] == "TOP_TO_BOTTOM"
    # tall line inside a wide caption region -> horizontal (region wins)
    para = build_paragraph(
        [{"text": "x", "xyxy": (100, 800, 130, 900)}],
        1351,
        1920,
        region_xyxy=(5, 805, 1210, 905),
    )
    assert para is not None
    assert para["writing_direction"] == "LEFT_TO_RIGHT"


def test_single_line_without_region_falls_back_to_merged_aspect():
    para = build_paragraph([{"text": "あ", "xyxy": (0, 0, 20, 100)}], 200, 200)
    assert para is not None
    assert para["writing_direction"] == "TOP_TO_BOTTOM"
    para = build_paragraph([{"text": "a", "xyxy": (0, 0, 100, 20)}], 200, 200)
    assert para is not None
    assert para["writing_direction"] == "LEFT_TO_RIGHT"


def test_build_paragraph_skips_empty():
    assert build_paragraph([{"text": "   ", "xyxy": (0, 0, 10, 10)}], 100, 100) is None
    assert build_paragraph([], 100, 100) is None


def test_paragraph_merges_boxes():
    para = build_paragraph(
        [
            {"text": "a", "xyxy": (10, 10, 50, 30)},
            {"text": "b", "xyxy": (60, 40, 90, 60)},
        ],
        100,
        100,
    )
    assert para is not None
    bb = para["bounding_box"]
    # merged box covers both lines: x 10..90, y 10..60
    assert bb["center_x"] == 0.5
    assert bb["center_y"] == 0.35
