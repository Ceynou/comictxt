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
    lines = [
        _ln("漢字の文章", (100, 100, 200, 400)),  # main, thickness 100
        _ln("かんじ", (204, 120, 244, 220)),  # thickness 40 > 32px cap -> kept
    ]
    assert len(filter_furigana(lines)) == 2


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
    assert cfg.lines.furigana_size_ratio == 0.70
