"""Group recognized lines into owocr-compatible paragraphs.

Convention (matches neokuro/converter.py):
- lines inside one YOLO region form one paragraph
- paragraph writing_direction is layout-based: vertical text lines are
  arranged side-by-side along x (with y-overlap); horizontal text lines
  are stacked along y. Single-line paragraphs fall back to the parent
  region-box aspect (never the line's own aspect: detector boxes are
  loose and short lines/SFX would misvote).

Reading order follows Kellenok's PP-OCR_manga Space app
(``sort_reading_order``): vertical text reads column by column, right to
left, top to bottom within a column; horizontal text reads row by row,
top to bottom, left to right within a row. Orientation comes from the
Space's area-weighted vote.
"""
from __future__ import annotations

from comictxt.geometry import merge_boxes_to_xyxy, normalized_bbox_dict


def _y_overlap(a: tuple, b: tuple) -> float:
    inter = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / max(1.0, min(a[3] - a[1], b[3] - b[1]))


def _x_overlap(a: tuple, b: tuple) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    return inter / max(1.0, min(a[2] - a[0], b[2] - b[0]))


def _h_gap(a: tuple, b: tuple) -> float:
    if a[2] < b[0]:
        return b[0] - a[2]
    if b[2] < a[0]:
        return a[0] - b[2]
    return 0.0


def _v_gap(a: tuple, b: tuple) -> float:
    if a[3] < b[1]:
        return b[1] - a[3]
    if b[3] < a[1]:
        return a[1] - b[3]
    return 0.0


def infer_orientation(
    boxes: list[tuple[float, float, float, float]],
    region_xyxy: tuple[float, float, float, float] | None = None,
    layout_overlap_ratio: float = 0.5,
    layout_gap_ratio: float = 0.75,
) -> bool:
    """Return True for vertical (TOP_TO_BOTTOM), False for horizontal.

    Adjacent-pair majority (chimahon-style): vertical lines are horizontally
    adjacent (small x-gap vs line width + strong y-overlap); horizontal lines
    are vertically adjacent (small y-gap + strong x-overlap). Single line or
    no majority: parent region-box aspect, then merged-box aspect.
    """
    if len(boxes) >= 2:
        by_x = sorted(boxes, key=lambda b: (b[0] + b[2]) / 2.0)
        v_good = 0
        for a, b in zip(by_x, by_x[1:]):
            w = ((a[2] - a[0]) + (b[2] - b[0])) / 2.0
            if _h_gap(a, b) < w * layout_gap_ratio and _y_overlap(a, b) >= layout_overlap_ratio:
                v_good += 1
        by_y = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2.0)
        h_good = 0
        for a, b in zip(by_y, by_y[1:]):
            h = max(a[3] - a[1], b[3] - b[1])
            if _v_gap(a, b) < h * 1.5 and _x_overlap(a, b) >= layout_overlap_ratio:
                h_good += 1
        pairs = len(boxes) - 1
        v_maj = v_good * 2 >= pairs
        h_maj = h_good * 2 >= pairs
        if v_maj and not h_maj:
            return True
        if h_maj and not v_maj:
            return False
        if v_maj and h_maj:
            # Close call (staggered/tilted lines look adjacent both ways):
            # defer to the parent region aspect instead of trusting a
            # one-pair margin from inflated axis boxes.
            if abs(v_good - h_good) <= max(1, pairs // 4):
                pass  # fall through to region aspect below
            else:
                return v_good >= h_good
        # no majority: fall through to region aspect
    if region_xyxy is not None:
        x1, y1, x2, y2 = region_xyxy
        return (y2 - y1) > (x2 - x1)
    # last resort: merged-box aspect
    px1, py1, px2, py2 = merge_boxes_to_xyxy(boxes)
    return (py2 - py1) > (px2 - px1)


def build_paragraph(
    lines: list[dict],
    W: int,
    H: int,
    region_xyxy: tuple[float, float, float, float] | None = None,
    layout_overlap_ratio: float = 0.5,
    layout_gap_ratio: float = 0.75,
) -> dict | None:
    """lines: [{'text': str, 'xyxy': (x1,y1,x2,y2)}]. Returns owocr paragraph or None.

    Items may carry ``sort_wh`` (deskewed (w, h) in image px, from the
    detector quad): axis boxes of tilted lines are inflated, which merges
    distinct columns/rows in the Space sorter. The output ``bounding_box``
    always stays axis-aligned; ``sort_wh`` only steers reading order.
    """
    kept = [ln for ln in lines if ln.get("text", "").strip()]
    if not kept:
        return None
    boxes = [ln["xyxy"] for ln in kept]
    px1, py1, px2, py2 = merge_boxes_to_xyxy(boxes)
    vertical = infer_orientation(boxes, region_xyxy, layout_overlap_ratio, layout_gap_ratio)
    line_objs = []
    for ln in kept:
        x1, y1, x2, y2 = ln["xyxy"]
        bbox = normalized_bbox_dict(x1, y1, x2, y2, W, H)
        obj: dict = {
            "text": ln["text"],
            "bounding_box": bbox,
            "words": [{"text": ln["text"], "bounding_box": dict(bbox)}],
        }
        swh = ln.get("sort_wh")
        if swh is not None:
            try:
                sw, sh = float(swh[0]) / max(1, W), float(swh[1]) / max(1, H)
            except (TypeError, ValueError):
                sw, sh = 0.0, 0.0
            if sw > 0 and sh > 0:
                obj["_sort_wh"] = [sw, sh]
        line_objs.append(obj)
    return {
        "bounding_box": normalized_bbox_dict(px1, py1, px2, py2, W, H),
        "lines": line_objs,
        "writing_direction": "TOP_TO_BOTTOM" if vertical else "LEFT_TO_RIGHT",
    }


def _is_vertical_or_rtl(paragraph: dict) -> bool:
    """True for vertical/RTL blocks, False for LEFT_TO_RIGHT."""
    return paragraph.get("writing_direction", "LEFT_TO_RIGHT") != "LEFT_TO_RIGHT"


def _norm_box(obj: dict) -> dict:
    """Normalized bounding_box -> Space-style box dict."""
    bb = obj.get("bounding_box", {})
    cx = float(bb.get("center_x", 0.5))
    cy = float(bb.get("center_y", 0.5))
    w = float(bb.get("width", 0.0))
    h = float(bb.get("height", 0.0))
    return {
        "xmin": cx - w / 2.0, "xmax": cx + w / 2.0,
        "ymin": cy - h / 2.0, "ymax": cy + h / 2.0,
        "cx": cx, "cy": cy,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def vote_vertical_boxes(boxes: list[dict]) -> bool:
    """Space area-weighted orientation vote over Space-style box dicts."""
    vert_weight = 0.0
    horiz_weight = 0.0
    for b in boxes:
        bw = b["xmax"] - b["xmin"]
        bh = b["ymax"] - b["ymin"]
        area = bw * bh
        if bh > bw * 1.1:
            vert_weight += area
        elif bw > bh * 1.1:
            horiz_weight += area
    return bool(vert_weight >= horiz_weight)


def sort_reading_order(items: list, box_of, is_vertical: bool = True) -> list:
    """Space ``sort_reading_order`` ported to generic items.

    ``box_of(item)`` returns a Space-style box dict. Vertical: columns
    right-to-left, lines top-to-bottom within a column. Horizontal: rows
    top-to-bottom, items left-to-right within a row. Column/row membership
    threshold is 0.60 of the mean width/height (Space verbatim).
    """
    if not items:
        return []
    if is_vertical:
        ordered = sorted(items, key=lambda it: -box_of(it)["cx"])
        columns: list[list] = []
        for it in ordered:
            b = box_of(it)
            placed = False
            for col in columns:
                col_cx = _mean([box_of(cb)["cx"] for cb in col])
                col_w = _mean([box_of(cb)["xmax"] - box_of(cb)["xmin"] for cb in col])
                b_w = b["xmax"] - b["xmin"]
                if abs(b["cx"] - col_cx) < max(col_w, b_w) * 0.60:
                    col.append(it)
                    placed = True
                    break
            if not placed:
                columns.append([it])
        columns.sort(key=lambda col: -_mean([box_of(cb)["cx"] for cb in col]))
        result = []
        for col in columns:
            col.sort(key=lambda cb: box_of(cb)["ymin"])
            result.extend(col)
        return result
    else:
        ordered = sorted(items, key=lambda it: box_of(it)["cy"])
        rows: list[list] = []
        for it in ordered:
            b = box_of(it)
            placed = False
            for row in rows:
                r_cy = _mean([box_of(rb)["cy"] for rb in row])
                r_h = _mean([box_of(rb)["ymax"] - box_of(rb)["ymin"] for rb in row])
                b_h = b["ymax"] - b["ymin"]
                if abs(b["cy"] - r_cy) < max(r_h, b_h) * 0.60:
                    row.append(it)
                    placed = True
                    break
            if not placed:
                rows.append([it])
        rows.sort(key=lambda row: _mean([box_of(rb)["cy"] for rb in row]))
        result = []
        for row in rows:
            row.sort(key=lambda rb: box_of(rb)["xmin"])
            result.extend(row)
        return result


def reorder_paragraphs(paragraphs: list[dict]) -> list[dict]:
    """Order paragraphs in reading order (Space column/row sort).

    Global orientation comes from the area-weighted vote over paragraph
    boxes, exactly like the Space app votes over detected line boxes.
    """
    if len(paragraphs) < 2:
        return list(paragraphs)
    boxes = [_norm_box(p) for p in paragraphs]
    vertical = vote_vertical_boxes(boxes)
    by_box = {id(p): b for p, b in zip(paragraphs, boxes)}
    return sort_reading_order(
        list(paragraphs), box_of=lambda p: by_box[id(p)], is_vertical=vertical)


def _sort_box(line: dict) -> dict:
    """Space-style box for reading-order sorting of a paragraph line.

    Uses the deskewed ``_sort_wh`` size when present (center still comes
    from the axis-aligned ``bounding_box`` — rotation-invariant for
    parallelograms — but tilted lines' inflated axis w/h would merge
    distinct columns/rows).
    """
    b = _norm_box(line)
    swh = line.get("_sort_wh")
    if swh is not None:
        try:
            sw, sh = float(swh[0]), float(swh[1])
        except (TypeError, ValueError):
            sw, sh = 0.0, 0.0
        if sw > 0 and sh > 0:
            b = {
                "xmin": b["cx"] - sw / 2.0, "xmax": b["cx"] + sw / 2.0,
                "ymin": b["cy"] - sh / 2.0, "ymax": b["cy"] + sh / 2.0,
                "cx": b["cx"], "cy": b["cy"],
            }
    return b


def reorder_lines_in_paragraph(paragraph: dict) -> dict:
    """Sort a paragraph's lines in reading order (in place, also returned).

    Space column/row sort using the paragraph's own orientation, so
    staggered multi-column lines order correctly (not just a plain sort).
    """
    lines = paragraph.get("lines", [])
    if len(lines) < 2:
        return paragraph
    vertical = _is_vertical_or_rtl(paragraph)
    by_box = {id(ln): _sort_box(ln) for ln in lines}
    ordered = sort_reading_order(
        list(lines), box_of=lambda ln: by_box[id(ln)], is_vertical=vertical)
    lines[:] = ordered
    return paragraph


def apply_reading_order(
    paragraphs: list[dict],
    reorder_blocks: bool = True,
    reorder_lines: bool = True,
) -> list[dict]:
    """Apply end-of-pipeline reading-order sorting. Returns the same list."""
    if reorder_lines:
        for para in paragraphs:
            reorder_lines_in_paragraph(para)
    if reorder_blocks:
        return reorder_paragraphs(paragraphs)
    return paragraphs
