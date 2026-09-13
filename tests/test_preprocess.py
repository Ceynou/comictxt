"""Preprocessing (levels + sharpen) tests."""
import numpy as np

from comictxt.preprocess import apply_levels, apply_sharpen, build_levels_lut, cleanup


def test_lut_identity_at_defaults():
    lut = build_levels_lut(0, 255)
    assert lut[0] == 0 and lut[255] == 255
    assert (lut == np.arange(256, dtype=np.uint8)).all()


def test_levels_stretch():
    img = np.zeros((1, 4, 3), dtype=np.uint8)
    img[0, :, 0] = [0, 64, 128, 255]
    out = apply_levels(img, black_point=64, white_point=192)
    assert out[0, 0, 0] == 0  # below black point -> 0
    assert out[0, 3, 0] == 255  # above white point -> 255
    assert out[0, 1, 0] == 0  # at black point -> 0
    mid = out[0, 2, 0]
    assert 100 < mid < 160  # 128 stretched between


def test_levels_identity_passthrough():
    img = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    assert apply_levels(img, 0, 255) is img


def test_sharpen_identity_at_zero():
    img = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    assert apply_sharpen(img, 0.0) is img


def test_sharpen_changes_edges():
    # Mid-gray step (no clipping) so the unsharp mask visibly changes pixels.
    img = np.full((16, 16, 3), 100, np.uint8)
    img[:, 8:] = 155
    out = apply_sharpen(img, 1.0)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out, img)


def test_cleanup_disabled_passthrough():
    img = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    assert cleanup(img, enable=False) is img


def test_cleanup_enabled_applies_both():
    img = np.zeros((16, 16, 3), np.uint8)
    img[:, 8:] = 200
    out = cleanup(img, enable=True, black_point=50, white_point=150, sharpen=0.5)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_pipeline_cleanup_pil_passthrough():
    from PIL import Image

    from comictxt.config import ComictxtConfig
    from comictxt.pipeline import ComicTxtPipeline

    pipe = ComicTxtPipeline(ComictxtConfig())
    img = Image.new("RGB", (10, 10), (128, 128, 128))
    assert pipe._cleanup_pil(img) is img
