"""Furigana tests for the Kellenok Space port (is_furigana_pair rules)."""
from comictxt.furigana import contains_kanji, filter_furigana, is_furigana_pair


def _ln(text, xyxy):
    return {"text": text, "xyxy": xyxy}


def test_contains_kanji():
    assert contains_kanji("漢字") is True
    assert contains_kanji("かんじ") is False
    assert contains_kanji("Hello") is False


def test_ruby_right_of_vertical_main_is_dropped():
    lines = [
        _ln("漢字の文章", (100, 100, 140, 300)),  # main, thickness 40
        _ln("かんじ", (142, 120, 158, 160)),  # small ruby right of main
    ]
    kept = filter_furigana(lines)
    assert [ln["text"] for ln in kept] == ["漢字の文章"]


def test_ruby_above_horizontal_main_is_dropped():
    lines = [
        _ln("漢字の文章です", (100, 120, 400, 160)),  # main, thickness 40
        _ln("かんじ", (120, 100, 200, 116)),  # ruby above main, thickness 16
    ]
    kept = filter_furigana(lines)
    assert [ln["text"] for ln in kept] == ["漢字の文章です"]


def test_thick_kana_kept():
    lines = [
        _ln("漢字", (100, 100, 140, 300)),
        _ln("ひらがな", (142, 120, 182, 320)),  # same thickness 40 -> not ruby
    ]
    assert len(filter_furigana(lines)) == 2


def test_absolute_thickness_cap():
    # Ruby that is proportionally small but physically large (hi-res scans,
    # big-font pages) is still ruby: the absolute cap defers to the relative
    # bound instead of keeping oversized ruby lines.
    lines = [
        _ln("漢字の文章", (100, 100, 200, 400)),  # main, thickness 100
        _ln("かんじ", (204, 120, 244, 220)),  # thickness 40 (raw) but ~0.38 of main
    ]
    assert [ln["text"] for ln in filter_furigana(lines)] == ["漢字の文章"]


def test_thick_parallel_kana_kept():
    # A kana line nearly as thick as its neighbor is parallel dialogue,
    # not ruby — the ratio law keeps it regardless of absolute size.
    lines = [
        _ln("漢字の文章", (100, 100, 200, 400)),  # thickness 100
        _ln("かんじ", (204, 120, 284, 320)),  # thickness 80 > 0.75 * main
    ]
    assert len(filter_furigana(lines)) == 2


def test_katakana_ruby_with_chouonpu():
    # Katakana ruby readings legitimately contain the long-vowel mark
    # (エキスパート over 熟練者！); pure ー marks stay kept.
    lines = [
        _ln("熟練者！", (541, 700, 562, 787)),
        _ln("エキスパート", (563, 699, 576, 767)),
    ]
    assert [ln["text"] for ln in filter_furigana(lines)] == ["熟練者！"]
    lines2 = [
        _ln("熟練者！", (541, 700, 562, 787)),
        _ln("ーー", (563, 699, 572, 767)),
    ]
    assert len(filter_furigana(lines2)) == 2


def test_far_kana_kept():
    lines = [
        _ln("漢字の文章", (100, 100, 140, 300)),
        _ln("かんじ", (300, 120, 316, 160)),  # too far -> not ruby
    ]
    assert len(filter_furigana(lines)) == 2


def test_kanji_lines_never_dropped():
    lines = [
        _ln("漢字", (100, 100, 140, 300)),
        _ln("日本語", (142, 120, 158, 160)),
    ]
    assert len(filter_furigana(lines)) == 2


def test_main_without_kanji_has_no_furigana():
    # Space rule 2: ruby annotates kanji — kanji-free main keeps neighbors.
    lines = [
        _ln("ひらがなのぶんしょう", (100, 100, 140, 300)),
        _ln("かんじ", (142, 120, 158, 160)),
    ]
    assert len(filter_furigana(lines)) == 2


def test_punctuation_veto():
    lines = [
        _ln("漢字の文章", (100, 100, 140, 300)),
        _ln("かんじ！", (142, 120, 158, 160)),  # dialogue punct -> never ruby
    ]
    assert len(filter_furigana(lines)) == 2


def test_long_sub_kept():
    lines = [
        _ln("漢字の文章", (100, 100, 140, 300)),
        _ln("かんじかんじかんじか", (142, 120, 158, 200)),  # >8 chars -> kept
    ]
    assert len(filter_furigana(lines)) == 2


def test_char_size_law_keeps_parallel_dialogue():
    # Equal char sizes with >=3 chars mean parallel dialogue, not ruby:
    # main 200px / 5 chars = 40px per char, sub 120px / 3 chars = 40px.
    lines = [
        _ln("漢字の文章", (100, 100, 140, 300)),
        _ln("かんじ", (142, 120, 158, 240)),
    ]
    assert len(filter_furigana(lines)) == 2


def test_wrong_side_kept():
    # Kana LEFT of vertical main (not right) is kept.
    lines = [
        _ln("漢字の文章", (100, 100, 140, 300)),
        _ln("かんじ", (60, 120, 76, 160)),
    ]
    assert len(filter_furigana(lines)) == 2


def test_is_furigana_pair_direct():
    sub = _ln("かんじ", (142, 120, 158, 160))
    main = _ln("漢字の文章", (100, 100, 140, 300))
    assert is_furigana_pair(sub, main, is_vertical=True) is True
    assert is_furigana_pair(sub, main, is_vertical=False) is False


def test_padded_crop_boxes_still_filter_via_det_xyxy():
    # Recognition crops carry rec.line_pad inflation, which flips the 0.70
    # thickness ratio for borderline ruby — filtering must use det_xyxy.
    lines = [
        {"text": "必殺技を覚え", "xyxy": (194.5, 632.9, 231.2, 858.6),
         "det_xyxy": (196.5, 634.9, 229.2, 856.6)},
        {"text": "おぼ", "xyxy": (229.3, 777.1, 255.8, 825.6),
         "det_xyxy": (231.3, 779.1, 253.8, 823.6)},
    ]
    assert [ln["text"] for ln in filter_furigana(lines)] == ["必殺技を覚え"]


def test_never_drops_last_line():
    lines = [_ln("あ", (0, 0, 10, 50))]
    assert filter_furigana(lines) == lines


def test_disabled_via_config():
    from comictxt.config import ComictxtConfig

    cfg = ComictxtConfig().with_overrides({"lines.furigana_filter": False})
    assert cfg.lines.furigana_filter is False
    assert cfg.lines.furigana_size_ratio == 0.75


def test_chunky_ruby_passes_via_ink_and_char_size():
    # Hand-drawn ruby can be ~0.85 of the main thickness; with ink sizes
    # present and the char-size law confirming ruby (chars ~half size),
    # the pair must still be recognized.
    sub = {"text": "わたし", "xyxy": (218, 650, 239, 685), "det_xyxy": (218, 650, 239, 685),
           "ink_wh": (21.0, 35.0)}
    main = {"text": "私は", "xyxy": (199, 652, 227, 703), "det_xyxy": (199, 652, 227, 703),
            "ink_wh": (23.0, 41.0)}
    assert is_furigana_pair(sub, main, is_vertical=True) is True


def test_ink_parallel_dialogue_kept():
    # Similar per-char size (>= 0.85 of main): parallel dialogue, not ruby,
    # even when the ink thickness ratio is small.
    sub = {"text": "そうです", "xyxy": (218, 650, 239, 750), "det_xyxy": (218, 650, 239, 750),
           "ink_wh": (21.0, 96.0)}
    main = {"text": "青い空", "xyxy": (199, 652, 227, 790), "det_xyxy": (199, 652, 227, 790),
            "ink_wh": (23.0, 69.0)}
    assert is_furigana_pair(sub, main, is_vertical=True) is False


def test_filter_uses_ink_wh():
    lines = [
        {"text": "必殺技を覚え", "xyxy": (196, 634, 230, 857), "det_xyxy": (196, 634, 230, 857),
         "ink_wh": (32.0, 222.0)},
        {"text": "ひっさつ", "xyxy": (233, 633, 253, 750), "det_xyxy": (233, 633, 253, 750),
         "ink_wh": (18.0, 109.0)},
    ]
    assert [ln["text"] for ln in filter_furigana(lines)] == ["必殺技を覚え"]
