"""Furigana (ruby text) filter.

Port of ``is_furigana_pair`` from Kellenok's PP-OCR_manga Space app: ruby
exists only to annotate kanji, sits strictly right of vertical (or above
horizontal) main text, and obeys physical scale laws (absolute 32px cap,
70% thickness, length law, char-size law). Runs post-recognition.

lines: [{'text': str, 'xyxy': (x1,y1,x2,y2)}] in image pixels.
Returns the filtered list (ruby lines removed).
"""
from __future__ import annotations

import re

_KANJI_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

# Dialogue punctuation can never be furigana (Space verbatim list).
_FURIGANA_FORBIDDEN_PUNCT = ("「", "」", "『", "』", "！", "？", "!", "?", "…", "。", "、", "―", "ー")


def contains_kanji(text: str) -> bool:
    """True if text contains any CJK ideograph/kanji (Space verbatim)."""
    return any(("\u4e00" <= ch <= "\u9fff") or ("\u3400" <= ch <= "\u4dbf") for ch in text)


def _as_box(xyxy: tuple) -> dict:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    return {
        "xmin": x1, "xmax": x2, "ymin": y1, "ymax": y2,
        "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
    }


def _filter_box(line: dict) -> tuple:
    """Box used for ruby geometry: the tight detection box when present.

    Recognition crops carry ``rec.line_pad`` inflation, which breaks the
    Space's thickness ratios for borderline ruby — the Space filters on
    tight detection boxes, so we do the same (``det_xyxy`` falls back to
    ``xyxy`` for region-text items that have no detection box).

    When the detector ``quad`` is present, the box is deskewed to its true
    size around the same center: axis boxes of tilted lines are inflated,
    which otherwise flips the thickness ratio for ruby beside tilted text.
    """
    quad = line.get("quad")
    if quad is not None:
        try:
            from comictxt.geometry import quad_true_size

            tw, th = quad_true_size(quad)
            base = line.get("det_xyxy", line["xyxy"])
            x1, y1, x2, y2 = (float(v) for v in base)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if tw > 0 and th > 0:
                return (cx - tw / 2.0, cy - th / 2.0, cx + tw / 2.0, cy + th / 2.0)
        except (TypeError, ValueError):
            pass
    return line.get("det_xyxy", line["xyxy"])


def vote_vertical(lines: list[dict], W=None, H=None) -> bool:
    """Area-weighted orientation vote over line boxes (Space verbatim).

    Boxes taller than 1.1x their width vote vertical (weighted by area) and
    vice versa; near-squares split their vote (Space: to the image's dominant
    axis, or abstain when image size is unknown). Votes on filter boxes.
    """
    vert_weight = 0.0
    horiz_weight = 0.0
    for ln in lines:
        b = _as_box(_filter_box(ln))
        bw = b["xmax"] - b["xmin"]
        bh = b["ymax"] - b["ymin"]
        area = bw * bh
        if bh > bw * 1.1:
            vert_weight += area
        elif bw > bh * 1.1:
            horiz_weight += area
        elif W is not None and H is not None:
            if H >= W:
                vert_weight += area * 0.5
            else:
                horiz_weight += area * 0.5
    return bool(vert_weight >= horiz_weight)


def is_furigana_pair(
    sub_line: dict,
    main_line: dict,
    is_vertical: bool = True,
    size_ratio: float = 0.70,
    proximity_ratio: float = 0.35,
    overlap_ratio: float = 0.05,
    max_thickness_px: float = 32.0,
    length_ratio: float = 1.05,
    char_size_ratio: float = 0.85,
    proximity_min: float = 8.0,
    proximity_max: float = 16.0,
    max_chars: int = 8,
    box_pad: float = 4.0,
) -> bool:
    """Whether sub_line is ruby furigana belonging to main_line (Space port).

    ``box_pad`` compensates the detector quad's ``minAreaRect + box_pad``
    safety inflation before comparing thickness: the pad adds a constant to
    both lines, which inflates the sub/main thickness ratio for small ruby
    (e.g. real 0.67 reads as 0.75) and lets ruby escape the filter.
    """
    text_sub = (sub_line.get("text") or "").strip()
    text_main = (main_line.get("text") or "").strip()
    if not text_sub or not text_main:
        return False

    # Rule 1: furigana itself holds no kanji or dialogue punctuation, and is short.
    if contains_kanji(text_sub):
        return False
    if any(p in text_sub for p in _FURIGANA_FORBIDDEN_PUNCT if p != "ー"):
        return False
    elif "ー" in text_sub:
        # The long-vowel mark is forbidden dialogue punctuation, but real
        # katakana ruby readings contain it (エキスパート, ショー). Allow it
        # when the rest of the ruby is katakana; pure ー marks stay kept.
        stripped = text_sub.replace("ー", "")
        if not stripped or not all("\u30a0" <= ch <= "\u30ff" for ch in stripped):
            return False
    if len(text_sub) > max_chars:
        return False

    # Rule 2: furigana annotates KANJI — a kanji-free main line has none.
    if not contains_kanji(text_main):
        return False

    box_sub = _as_box(_filter_box(sub_line))
    box_main = _as_box(_filter_box(main_line))

    sub_w = box_sub["xmax"] - box_sub["xmin"]
    sub_h = box_sub["ymax"] - box_sub["ymin"]
    main_w = box_main["xmax"] - box_main["xmin"]
    main_h = box_main["ymax"] - box_main["ymin"]

    # Ink geometry: detector quads carry box_pad inflation and minAreaRect
    # slack; the measured ink extent (warp + Otsu bbox) is far more stable
    # for ruby ratios (det-box ratios wander 0.5-0.85 for true ruby).
    ink_sub = sub_line.get("ink_wh")
    ink_main = main_line.get("ink_wh")
    if ink_sub and ink_main:
        try:
            sub_thickness = float(min(ink_sub))
            main_thickness = float(min(ink_main))
            sub_len = float(max(ink_sub))
            main_len = float(max(ink_main))
            ink_ok = True
        except (TypeError, ValueError):
            ink_ok = False
    else:
        ink_ok = False
    if not ink_ok:
        sub_thickness = min(sub_w, sub_h)
        main_thickness = min(main_w, main_h)
        # Ink correction: detector quads carry a constant box_pad inflation
        # on every side dimension; thickness ratios on raw boxes are biased
        # up for small ruby (the pad is a bigger fraction of a thin line).
        if box_pad > 0:
            sub_thickness = max(1.0, sub_thickness - box_pad)
            main_thickness = max(1.0, main_thickness - box_pad)
        sub_len = max(sub_w, sub_h)
        main_len = max(main_w, main_h)

    # 4. Character size law FIRST: parallel dialogue shares char size, ruby
    #    is ~half. For >=2-char candidates this is the most reliable
    #    discriminator (some styles draw chunky ruby at ~0.85 thickness).
    char_size_confirmed = False
    if len(text_sub) >= 2:
        char_size_sub = sub_len / max(1, len(text_sub))
        char_size_main = main_len / max(1, len(text_main))
        if char_size_sub >= char_size_main * char_size_ratio:
            return False
        char_size_confirmed = True

    # Thickness law: strict by default; when the char-size law already
    # confirmed ruby, only require the ruby to not be FATTER than the main
    # (chunky hand-drawn ruby exists; dialogue is excluded by rule 4).
    eff_size_ratio = 0.95 if char_size_confirmed else size_ratio

    # 1. Physical scale limit: ruby stays small even in high-res scans.
    #    After the relative rule below, an absolute cap tighter than
    #    eff_size_ratio * main can only reject large-font pages' ruby, so
    #    the effective cap never drops below the relative bound.
    if sub_thickness > max(max_thickness_px, main_thickness * eff_size_ratio):
        return False

    # 2. Ruby is significantly thinner than the main line.
    if sub_thickness > main_thickness * eff_size_ratio:
        return False

    # 3. Length law: ruby annotates a word, never outgrows the main text.
    if sub_len > main_len * length_ratio:
        return False

    proximity_limit = min(proximity_max, max(proximity_min, main_thickness * proximity_ratio))

    if is_vertical:
        # Ruby sits strictly RIGHT of vertical main text.
        if box_sub["cx"] <= box_main["cx"]:
            return False
        oy = max(0.0, min(box_sub["ymax"], box_main["ymax"]) - max(box_sub["ymin"], box_main["ymin"]))
        y_overlap_ratio = oy / max(0.001, min(sub_h, main_h))
        if y_overlap_ratio < overlap_ratio:
            return False
        x_gap = max(0.0, box_sub["xmin"] - box_main["xmax"])
        return bool(x_gap <= proximity_limit)
    else:
        # Ruby sits strictly ABOVE horizontal main text.
        if box_sub["cy"] >= box_main["cy"]:
            return False
        ox = max(0.0, min(box_sub["xmax"], box_main["xmax"]) - max(box_sub["xmin"], box_main["xmin"]))
        x_overlap_ratio = ox / max(0.001, min(sub_w, main_w))
        if x_overlap_ratio < overlap_ratio:
            return False
        y_gap = max(0.0, box_main["ymin"] - box_sub["ymax"])
        return bool(y_gap <= proximity_limit)


def filter_furigana(
    lines: list[dict],
    size_ratio: float = 0.70,
    proximity_ratio: float = 0.35,
    overlap_ratio: float = 0.05,
    max_thickness_px: float = 32.0,
    length_ratio: float = 1.05,
    char_size_ratio: float = 0.85,
    proximity_min: float = 8.0,
    proximity_max: float = 16.0,
    max_chars: int = 8,
    is_vertical=None,
    box_pad: float = 4.0,
) -> list[dict]:
    """Drop ruby lines. Never drops the last remaining line."""
    kept, _events = explain_filter(
        lines,
        size_ratio=size_ratio,
        proximity_ratio=proximity_ratio,
        overlap_ratio=overlap_ratio,
        max_thickness_px=max_thickness_px,
        length_ratio=length_ratio,
        char_size_ratio=char_size_ratio,
        proximity_min=proximity_min,
        proximity_max=proximity_max,
        max_chars=max_chars,
        is_vertical=is_vertical,
        box_pad=box_pad,
    )
    return kept


def explain_filter(
    lines: list[dict],
    size_ratio: float = 0.70,
    proximity_ratio: float = 0.35,
    overlap_ratio: float = 0.05,
    max_thickness_px: float = 32.0,
    length_ratio: float = 1.05,
    char_size_ratio: float = 0.85,
    proximity_min: float = 8.0,
    proximity_max: float = 16.0,
    max_chars: int = 8,
    is_vertical=None,
    box_pad: float = 4.0,
) -> tuple[list[dict], list[dict]]:
    """Like filter_furigana but also returns per-line drop explanations.

    Returns ``(kept, events)`` where each event is
    ``{"index": int, "text": str, "dropped": bool, "reason": str,
    "main_index": int | None}``. ``is_vertical=None`` auto-votes orientation
    from the line boxes (Space area-weighted vote).
    """
    if len(lines) < 2:
        return list(lines), [
            {"index": i, "text": ln.get("text", ""), "dropped": False,
             "reason": "need >=2 lines", "main_index": None}
            for i, ln in enumerate(lines)
        ]
    vertical = vote_vertical(lines) if is_vertical is None else bool(is_vertical)
    drop = [False] * len(lines)
    events: list[dict] = [
        {"index": i, "text": ln.get("text", ""), "dropped": False,
         "reason": "kept", "main_index": None}
        for i, ln in enumerate(lines)
    ]
    for j, sub in enumerate(lines):
        matched = False
        for i, main in enumerate(lines):
            if i == j or drop[i]:
                continue
            if is_furigana_pair(
                sub, main,
                is_vertical=vertical,
                size_ratio=size_ratio,
                proximity_ratio=proximity_ratio,
                overlap_ratio=overlap_ratio,
                max_thickness_px=max_thickness_px,
                length_ratio=length_ratio,
                char_size_ratio=char_size_ratio,
                proximity_min=proximity_min,
                proximity_max=proximity_max,
                max_chars=max_chars,
                box_pad=box_pad,
            ):
                drop[j] = True
                matched = True
                events[j] = {
                    "index": j,
                    "text": sub.get("text", ""),
                    "dropped": True,
                    "reason": f"ruby of line {i} ({'vertical' if vertical else 'horizontal'})",
                    "main_index": i,
                }
                break
        if not matched:
            events[j]["reason"] = "kept: no kanji main line matched (Space rules)"
    kept = [ln for ln, d in zip(lines, drop) if not d]
    if not kept:
        for e in events:
            e["dropped"] = False
            e["reason"] += " (reverted: never drop last line)"
        return list(lines), events
    return kept, events
