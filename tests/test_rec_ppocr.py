"""PP-OCR recognizer tests: CTC decode unit + trim/warp geometry + model smoke."""
import numpy as np
import pytest

from comictxt.rec_ppocr import (
    ctc_greedy_decode,
    resolve_ppocr_dict,
    resolve_ppocr_model,
    trim_crop,
    warp_quad,
)

VOCAB = ["blank", "あ", "い", "う", " "]


def test_ctc_decode_skips_blanks_and_repeats():
    # Space-verbatim: a blank resets the repeat collapse.
    assert ctc_greedy_decode([0, 1, 1, 0, 2, 0, 2, 3, 3], VOCAB) == "あいいう"
    assert ctc_greedy_decode([0, 0, 0], VOCAB) == ""
    assert ctc_greedy_decode([1], VOCAB) == "あ"


def test_trim_crop_passthrough_degenerate():
    assert trim_crop(np.zeros((0, 0, 3), np.uint8)).size == 0
    wide = np.full((10, 100, 3), 255, np.uint8)
    assert trim_crop(wide) is wide  # horizontal white: trim guard keeps input


def test_warp_quad_axis_aligned():
    img = np.full((100, 100, 3), 255, np.uint8)
    quad = np.array([[10, 10], [50, 10], [50, 60], [10, 60]], dtype=np.float32)
    warped, new_quad = warp_quad(img, quad, trim=False)
    # tall warp is rotated to wide (Space behavior): 40x50 -> 50x40
    assert warped.shape == (40, 50, 3)
    assert new_quad.shape == (4, 2)


def test_warp_quad_degenerate():
    img = np.full((20, 20, 3), 255, np.uint8)
    quad = np.zeros((4, 2), dtype=np.float32)
    warped, _ = warp_quad(img, quad, trim=False)
    assert warped.shape == (10, 10, 3)


def test_resolvers_hit_local_cache():
    model = resolve_ppocr_model()
    assert model.is_file() and model.suffix == ".onnx"
    d = resolve_ppocr_dict()
    assert d.is_file() and d.name == "ppocrv6_dict.txt"


def test_resolvers_explicit_missing():
    from comictxt.rec_ppocr import resolve_ppocr_dict as _rd
    from comictxt.rec_ppocr import resolve_ppocr_model as _rm

    with pytest.raises(FileNotFoundError):
        _rm("/nonexistent/rec.onnx")
    with pytest.raises(FileNotFoundError):
        _rd("/nonexistent/dict.txt")


def test_real_model_blank_crop_decodes_empty():
    from comictxt.rec_ppocr import PpocrRecognizer

    rec = PpocrRecognizer(
        model_path=resolve_ppocr_model(), dict_path=resolve_ppocr_dict())
    rec.load()
    assert len(rec._vocab) > 100
    white = np.full((48, 200, 3), 255, np.uint8)
    assert rec._decode_bgr(white) == ""


def test_real_model_reads_synthetic_stripes():
    # Loose smoke test: high-contrast glyph-like bars must not crash and
    # should yield a string (content depends on the model).
    from comictxt.rec_ppocr import PpocrRecognizer

    rec = PpocrRecognizer(
        model_path=resolve_ppocr_model(), dict_path=resolve_ppocr_dict())
    rec.load()
    crop = np.full((48, 200, 3), 255, np.uint8)
    for x in range(10, 190, 20):
        crop[:, x:x + 8] = 0
    text = rec._decode_bgr(crop)
    assert isinstance(text, str)
