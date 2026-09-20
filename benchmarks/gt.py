"""Shared ground-truth loading for benchmarks.

Each GT page: mokuro-style JSON (blocks with box/vertical/lines/lines_coords)
paired with an image (same stem, any supported suffix).
"""
from __future__ import annotations

import json
from pathlib import Path

from comictxt.io_utils import load_pil

IMAGE_SUFFIXES = (".webp", ".png", ".jpg", ".jpeg", ".jxl")


def find_image(gt_path: Path) -> Path | None:
    for suf in IMAGE_SUFFIXES:
        cand = gt_path.with_suffix(suf)
        if cand.exists():
            return cand
    return None


def load_gt_pages(gt_dir: str | Path) -> list[dict]:
    """Returns [{"page": stem, "img": PIL.Image, "bgr": np.ndarray,
                 "blocks": [...], "lines": [{"quad", "text", "vertical",
                 "font_size", "block_idx"}]}]."""
    import numpy as np

    pages = []
    for gt_path in sorted(Path(gt_dir).glob("*.json")):
        img_path = find_image(gt_path)
        if img_path is None:
            continue
        data = json.loads(gt_path.read_text(encoding="utf-8"))
        img = load_pil(img_path).convert("RGB")
        bgr = np.array(img)[:, :, ::-1].copy()
        lines = []
        for bi, block in enumerate(data["blocks"]):
            coords = block.get("lines_coords") or []
            for text, quad in zip(block.get("lines", []), coords):
                lines.append({
                    "quad": quad,
                    "text": text,
                    "vertical": block.get("vertical", False),
                    "font_size": block.get("font_size"),
                    "block_idx": bi,
                })
        pages.append({
            "page": gt_path.stem,
            "img": img,
            "bgr": bgr,
            "blocks": data["blocks"],
            "lines": lines,
        })
    return pages


def quad_to_xyxy(quad: list) -> tuple[float, float, float, float]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return (min(xs), min(ys), max(xs), max(ys))
