"""Text-only recognition benchmark (Hayai OCR v2.5 Nova).

Recognizes GT line quads directly (no detection involved) and reports CER /
exact-match, split into text vs SFX buckets (font_size >= sfx_font_size).

Sweeps rec.max_num_patches x rec.line_pad; crops are prepared exactly like
ComicTxtPipeline._recognize_warped_quad (warp -> pad -> min-size upscale).

Usage:
  python benchmarks/bench_rec.py [--sweeps] [--limit N] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gt import load_gt_pages, quad_to_xyxy  # noqa: E402

from comictxt.eval import cer, normalize_text  # noqa: E402
from comictxt.geometry import warp_quad_to_rect  # noqa: E402


def make_crop(img_bgr: np.ndarray, quad: np.ndarray, line_pad: int, min_crop_size: int):
    warped = warp_quad_to_rect(img_bgr, quad)
    if warped.size == 0 or warped.shape[0] < 2 or warped.shape[1] < 2:
        return None
    if line_pad > 0:
        warped = cv2.copyMakeBorder(
            warped, line_pad, line_pad, line_pad, line_pad, cv2.BORDER_REPLICATE)
    if min_crop_size > 0 and (warped.shape[0] < min_crop_size or warped.shape[1] < min_crop_size):
        scale = max(min_crop_size / max(1, warped.shape[0]),
                    min_crop_size / max(1, warped.shape[1]))
        scale = min(scale, 8.0)
        warped = cv2.resize(warped, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return warped[:, :, ::-1]  # BGR -> RGB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="ground_truth")
    ap.add_argument("--sweeps", action="store_true", help="sweep patches x line_pad")
    ap.add_argument("--limit", type=int, default=0, help="recognize only first N lines (smoke)")
    ap.add_argument("--sfx-font-size", type=float, default=60.0,
                    help="font_size >= this counts as SFX bucket")
    ap.add_argument("--out", default="", help="write results JSON here")
    args = ap.parse_args()

    from comictxt.rec_hayai_torch import TorchHayaiRecognizer, resolve_rec_processor, resolve_rec_torch

    pages = load_gt_pages(args.gt_dir)
    lines = [(p["page"], ln) for p in pages for ln in p["lines"]]
    if args.limit:
        lines = lines[: args.limit]
    print(f"{len(pages)} pages, {len(lines)} GT lines")

    model = TorchHayaiRecognizer(
        model_path=resolve_rec_torch(""),
        processor_path=resolve_rec_processor(""),
        device="cpu",
        dtype="float32",
        max_new_tokens=128,
        max_num_patches=512,
    ).load()

    def run(line_pad: int, patches: int) -> dict:
        model.max_num_patches = patches
        rows = []
        t0 = time.time()
        for page, ln in lines:
            img_bgr = next(p["bgr"] for p in pages if p["page"] == page)
            crop = make_crop(img_bgr, np.asarray(ln["quad"], dtype=np.float32),
                             line_pad, 8)
            if crop is None:
                rows.append({"page": page, "ref": ln["text"], "hyp": "",
                             "cer": 1.0, "em": False,
                             "font_size": ln["font_size"], "vertical": ln["vertical"]})
                continue
            from PIL import Image

            hyp = model.ocr_pil(Image.fromarray(crop))
            ref = normalize_text(ln["text"])
            h = normalize_text(hyp)
            rows.append({"page": page, "ref": ln["text"], "hyp": hyp,
                         "cer": cer(ln["text"], hyp),
                         "em": ref == h and ref != "",
                         "font_size": ln["font_size"], "vertical": ln["vertical"]})
        wall = time.time() - t0
        return summarize(rows, wall, line_pad, patches)

    def summarize(rows: list, wall: float, line_pad: int, patches: int) -> dict:
        def bucket(rows, pred):
            sel = [r for r in rows if pred(r)]
            n = len(sel)
            return {
                "n": n,
                "cer": round(sum(r["cer"] for r in sel) / max(1, n), 4),
                "em": round(sum(r["em"] for r in sel) / max(1, n), 4),
            }

        fs = lambda r: r["font_size"] or 0  # noqa: E731
        return {
            "line_pad": line_pad,
            "max_num_patches": patches,
            "wall_s": round(wall, 1),
            "ms_per_line": round(wall * 1000 / max(1, len(rows)), 0),
            "all": bucket(rows, lambda r: True),
            "text": bucket(rows, lambda r: fs(r) < args.sfx_font_size),
            "sfx": bucket(rows, lambda r: fs(r) >= args.sfx_font_size),
            "rows": rows,
        }

    results = []
    if args.sweeps:
        grid = [(lp, p) for lp in (0, 2, 4) for p in (256, 384, 512)]
    else:
        grid = [(2, 512)]
    for line_pad, patches in grid:
        print(f"--- line_pad={line_pad} max_num_patches={patches}", flush=True)
        res = run(line_pad, patches)
        results.append(res)
        print(f"    all:  CER={res['all']['cer']:.4f} EM={res['all']['em']:.4f} "
              f"({res['all']['n']} lines, {res['ms_per_line']:.0f} ms/line)", flush=True)
        print(f"    text: CER={res['text']['cer']:.4f} EM={res['text']['em']:.4f} "
              f"({res['text']['n']})", flush=True)
        print(f"    sfx:  CER={res['sfx']['cer']:.4f} EM={res['sfx']['em']:.4f} "
              f"({res['sfx']['n']})", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
