"""Visualize line detections vs GT for one page (sanity-check FP buckets).

Usage: python benchmarks/viz_lines.py PAGE --scale 1280 --thresh 0.15 ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gt import load_gt_pages, quad_to_xyxy  # noqa: E402

from comictxt.config import resolve_lines_model  # noqa: E402
from comictxt.lines_ppocr import LineDetector  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("page")
    ap.add_argument("--gt-dir", default="ground_truth")
    ap.add_argument("--scale", type=int, default=1280)
    ap.add_argument("--thresh", type=float, default=0.15)
    ap.add_argument("--box-thresh", type=float, default=0.25)
    ap.add_argument("--unclip", type=float, default=1.4)
    ap.add_argument("--min-line-px", type=float, default=12.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pages = load_gt_pages(args.gt_dir)
    page = next(p for p in pages if p["page"] == args.page)
    det = LineDetector(
        model_path=resolve_lines_model(""), det_long_side=args.scale,
        thresh=args.thresh, box_thresh=args.box_thresh,
        unclip_ratio=args.unclip,
    ).load()
    quads = det.detect_pil(page["img"])
    W, H = page["img"].size
    img = page["img"].copy()
    d = ImageDraw.Draw(img)
    for ln in page["lines"]:
        x1, y1, x2, y2 = quad_to_xyxy(ln["quad"])
        d.rectangle([x1, y1, x2, y2], outline="blue", width=3)
    from bench_lines import greedy_match, looks_like_ruby  # noqa: E402

    det_boxes = []
    for q in quads:
        q = np.asarray(q, dtype=np.float32)
        x1, y1, x2, y2 = (float(q[:, 0].min()), float(q[:, 1].min()),
                          float(q[:, 0].max()), float(q[:, 1].max()))
        if args.min_line_px > 0 and max(x2 - x1, y2 - y1) < args.min_line_px:
            continue
        det_boxes.append((x1, y1, x2, y2))
    gt_boxes = [quad_to_xyxy(ln["quad"]) for ln in page["lines"]]
    matches = greedy_match(det_boxes, gt_boxes, 0.5)
    matched = {m[1] for m in matches}
    mains = [det_boxes[i] for i in matched] + list(gt_boxes)
    for i, b in enumerate(det_boxes):
        if i in matched:
            d.rectangle(b, outline="green", width=2)
        elif looks_like_ruby(b, mains, None):
            d.rectangle(b, outline="orange", width=2)
        else:
            d.rectangle(b, outline="red", width=2)
    print("blue=GT green=TP orange=ruby-like FP red=other FP;"
          f" {len(det_boxes)} dets, {len(gt_boxes)} gt")
    out = args.out or f"/tmp/opencode/viz_{args.page}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(out)


if __name__ == "__main__":
    main()
