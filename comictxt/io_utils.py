"""Image IO helpers."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

try:
    import pillow_jxl  # noqa: F401  (registers the .jxl handler on import)
    _JXL_AVAILABLE = True
except ImportError:
    _JXL_AVAILABLE = False


def _ensure_jxl(suffix: str) -> None:
    if suffix == ".jxl" and not _JXL_AVAILABLE:
        raise ImportError(
            "JPEG-XL support requires the 'pillow-jxl-plugin' package "
            "(pip install pillow-jxl-plugin)"
        )


def jxl_available() -> bool:
    """True when the JPEG-XL Pillow plugin is importable."""
    return _JXL_AVAILABLE


def load_pil(image: Union[str, Path, bytes, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        _ensure_jxl(Path(image).suffix.lower())
        return Image.open(str(image)).convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(image))).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def pil_to_cv2_bgr(pil: Image.Image) -> np.ndarray:
    arr = np.array(pil)  # RGB
    return arr[:, :, ::-1].copy()  # BGR


def image_size(image: Union[str, Path, bytes, Image.Image]) -> tuple[int, int]:
    return load_pil(image).size  # (W, H)
