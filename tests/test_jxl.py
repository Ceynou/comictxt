"""JPEG-XL input support tests (skipped when pillow-jxl-plugin is missing)."""
import pytest

from comictxt.cli import IMAGE_SUFFIXES
from comictxt.eval import IMAGE_SUFFIXES as EVAL_SUFFIXES
from comictxt.io_utils import jxl_available, load_pil

needs_jxl = pytest.mark.skipif(not jxl_available(), reason="pillow-jxl-plugin not installed")


def test_suffixes_include_jxl():
    assert ".jxl" in IMAGE_SUFFIXES
    assert ".jxl" in EVAL_SUFFIXES


def test_jxl_available_flag_matches_import():
    try:
        import pillow_jxl  # noqa: F401

        assert jxl_available() is True
    except ImportError:
        assert jxl_available() is False


@needs_jxl
def test_load_jxl_file_roundtrip(tmp_path):
    from PIL import Image

    src = Image.new("RGB", (64, 48), (10, 200, 90))
    p = tmp_path / "img.jxl"
    src.save(p)
    out = load_pil(p)
    assert out.size == (64, 48)
    assert out.mode == "RGB"
    px = out.getpixel((0, 0))
    assert px[1] > 150 and px[0] < 100  # lossy JXL: green dominant, roughly kept


@needs_jxl
def test_load_jxl_bytes_roundtrip(tmp_path):
    import io

    from PIL import Image

    src = Image.new("RGB", (32, 32), (200, 20, 20))
    p = tmp_path / "img.jxl"
    src.save(p)
    out = load_pil(p.read_bytes())
    assert out.size == (32, 32)


@needs_jxl
def test_eval_find_image_prefers_jxl(tmp_path):
    from comictxt.eval import find_image

    (tmp_path / "page.json").write_text("{}")
    (tmp_path / "page.jxl").write_bytes(b"")  # only name resolution matters
    assert find_image(tmp_path / "page.json") == tmp_path / "page.jxl"
