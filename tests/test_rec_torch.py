import pytest

from comictxt.config import ComictxtConfig
from comictxt.rec_hayai_torch import TorchHayaiRecognizer, resolve_rec_torch


def test_resolve_torch_snapshot():
    p = resolve_rec_torch("")
    assert (p / "model.safetensors").exists()
    assert (p / "tokenizer.json").exists()


def test_torch_config_defaults():
    cfg = ComictxtConfig()
    assert cfg.rec.backend == "ppocr"
    cfg2 = cfg.with_overrides({"rec.backend": "onnx", "rec.device": "cpu"})
    assert cfg2.rec.backend == "onnx"


def test_torch_recognizer_interface():
    rec = TorchHayaiRecognizer(model_path=resolve_rec_torch(""))
    assert hasattr(rec, "ocr_pil") and hasattr(rec, "ocr_array")
    assert hasattr(rec, "ensure_loaded")


@pytest.mark.slow
def test_torch_transcribes_synthetic_crop():
    from PIL import Image, ImageDraw

    rec = TorchHayaiRecognizer(model_path=resolve_rec_torch("")).load()
    img = Image.new("RGB", (300, 80), (255, 255, 255))
    ImageDraw.Draw(img).text((20, 20), "Hello", fill=(0, 0, 0))
    text = rec.ocr_pil(img)
    assert isinstance(text, str) and "Hello" in text


def test_resolve_torch_revision_pinning(monkeypatch):
    """The resolver must pin the requested revision, not the newest snapshot.

    Downloading another branch (e.g. nova-alpha) used to silently re-pin
    the default model via the mtime heuristic.
    """
    from pathlib import Path

    base = Path(resolve_rec_torch(""))  # any cached snapshot root
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    snap = resolve_rec_torch("", revision="nova-alpha")
    assert snap.parent.parent == base.parent.parent  # same repo
    ref = (snap.parent.parent / "refs" / "nova-alpha").read_text().strip()
    assert snap.name == ref
    # unknown revision falls back to the newest complete snapshot
    fallback = resolve_rec_torch("", revision="no-such-branch")
    assert (fallback / "model.safetensors").exists()
