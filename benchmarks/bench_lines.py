"""Line-detection benchmark (PP-OCRv6 manga DB detector) against GT lines.

Two detection contexts:
  gt-crop  — per GT block crop with pipeline region padding (mirrors the
             per-region pass with perfect regions)
  full     — full page (mirrors the orphan sweep)

Caches the DB probability map per (item, scale) so the post-processing sweep
(thresh / box_thresh / unclip_ratio / min_short_side / box_pad) is cheap.

Unmatched detections are bucketed into ruby-like (furigana-shaped: small,
beside a bigger aligned line) vs other false positives, so "line detection
without furigana" can be judged both ways.

Usage:
  python benchmarks/bench_lines.py [--stage main|refine] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gt import load_gt_pages, quad_to_xyxy  # noqa: E402

from comictxt.config import resolve_lines_model  # noqa: E402
from comictxt.geometry import expand_region_boxes  # noqa: E402
from comictxt.lines_ppocr import LineDetector  # noqa: E402

PAD_RATIO = 0.15
PAD_MODE = "auto"


def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def greedy_match(det_boxes, gt_boxes, iou_thresh=0.5):
    cands = []
    for i, d in enumerate(det_boxes):
        for j, g in enumerate(gt_boxes):
            iou = box_iou(d, g)
            if iou >= iou_thresh:
                cands.append((iou, i, j))
    cands.sort(reverse=True)
    used_d, used_g, matches = set(), set(), []
    for iou, i, j in cands:
        if i not in used_d and j not in used_g:
            used_d.add(i)
            used_g.add(j)
            matches.append((iou, i, j))
    return matches


def looks_like_ruby(det, mains, vertical_hint=None) -> bool:
    """Box-only ruby heuristic: small, thin, beside a bigger aligned line."""
    dx1, dy1, dx2, dy2 = det
    dw, dh = dx2 - dx1, dy2 - dy1
    d_len, d_thick = max(dw, dh), min(dw, dh)
    for m in mains:
        mx1, my1, mx2, my2 = m
        mw, mh = mx2 - mx1, my2 - my1
        m_len, m_thick = max(mw, mh), min(mw, mh)
        if d_thick > m_thick * 0.80 or d_len > m_len * 1.10:
            continue
        dcx, dcy = (dx1 + dx2) / 2, (dy1 + dy2) / 2
        mcx, mcy = (mx1 + mx2) / 2, (my1 + my2) / 2
        vertical = vertical_hint
        if vertical is None:
            vertical = mh > mw * 1.1
        if vertical:
            # ruby right of the main column, y-overlapping
            if dcx <= mcx:
                continue
            oy = min(dy2, my2) - max(dy1, my1)
            if oy < 0.05 * min(dh, mh):
                continue
            gap = dx1 - mx2
            if -d_thick <= gap <= max(16.0, 0.6 * m_thick):
                return True
        else:
            # ruby above the main row, x-overlapping
            if dcy >= mcy:
                continue
            ox = min(dx2, mx2) - max(dx1, mx1)
            if ox < 0.05 * min(dw, mw):
                continue
            gap = my1 - dy2
            if -d_thick <= gap <= max(16.0, 0.6 * m_thick):
                return True
    return False


class Ctx:
    """Detection context items: (page_idx, crop_origin, bgr, gt_line_boxes)."""

    def __init__(self, name: str):
        self.name = name
        self.items = []  # {"page": int, "origin": (x,y), "bgr": arr, "gt": [xyxy...], "vertical": bool|None}
        self.cache = {}  # (forward-key, item_idx) -> (pred_map, W, H, pW, pH, tw, th)


def build_contexts(pages, kind: str) -> Ctx:
    ctx = Ctx(kind)
    for pi, p in enumerate(pages):
        W, H = p["img"].size
        page_gt = [quad_to_xyxy(ln["quad"]) for ln in p["lines"]]
        if kind == "full":
            ctx.items.append({"page": pi, "origin": (0.0, 0.0), "bgr": p["bgr"],
                              "gt": page_gt, "vertical": None})
        else:  # gt-crop: one detection input per block, evaluated per page
            boxes = [tuple(float(v) for v in b["box"]) for b in p["blocks"]]
            padded = expand_region_boxes(boxes, PAD_RATIO, PAD_MODE, W, H)
            for block, pb in zip(p["blocks"], padded):
                x1, y1, x2, y2 = (int(v) for v in pb)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                ctx.items.append({
                    "page": pi,
                    "origin": (float(x1), float(y1)),
                    "bgr": p["bgr"][y1:y2, x1:x2].copy(),
                    "gt": page_gt,  # page-level GT; dedup happens per page
                    "vertical": block.get("vertical"),
                })
    return ctx


def _dedup_boxes(boxes: list, iou_thresh: float = 0.6) -> list:
    """Drop near-duplicate boxes (overlapping crops double-detect lines)."""
    out: list = []
    for b in boxes:
        if not any(box_iou(b, o) > iou_thresh or _center_in(b, o) for o in out):
            out.append(b)
    return out


def _center_in(a, b) -> bool:
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def forwards(ctx: Ctx, det: LineDetector) -> None:
    key = (det.det_long_side, det.det_min_side, det.det_margin)
    for i, item in enumerate(ctx.items):
        if (key, i) not in ctx.cache:
            ctx.cache[(key, i)] = det._forward(item["bgr"])


def detect(ctx: Ctx, det: LineDetector, min_line_px: float) -> list[dict]:
    """Run post-processing on cached maps; group detections per page.

    gt-crop unions + dedups the per-block detections per page (overlapping
    crops double-detect lines); full has one input per page already."""
    key = (det.det_long_side, det.det_min_side, det.det_margin)
    per_page: dict = {}
    for i, item in enumerate(ctx.items):
        pred_map, W, H, pW, pH, tw, th = ctx.cache[(key, i)]
        info = det._postprocess(pred_map, W, H, pW, pH, tw, th)
        ox, oy = item["origin"]
        boxes = per_page.setdefault(item["page"], [])
        for k in info["kept"]:
            q = np.asarray(k["quad"], dtype=np.float32)
            q[:, 0] += ox
            q[:, 1] += oy
            x1, y1, x2, y2 = float(q[:, 0].min()), float(q[:, 1].min()), \
                float(q[:, 0].max()), float(q[:, 1].max())
            if min_line_px > 0 and max(x2 - x1, y2 - y1) < min_line_px:
                continue
            boxes.append((x1, y1, x2, y2))
    out = []
    for pi in sorted(per_page):
        page = next(it for it in ctx.items if it["page"] == pi)
        dets = per_page[pi]
        if ctx.name == "gt-crop":
            dets = _dedup_boxes(dets)
        out.append({"gt": page["gt"], "dets": dets, "vertical": None})
    return out


def evaluate(results: list[dict], iou_thresh: float = 0.5) -> dict:
    tp = fp = fn = 0
    ruby_fp = other_fp = 0
    ious = []
    for r in results:
        matches = greedy_match(r["dets"], r["gt"], iou_thresh)
        tp += len(matches)
        fn += len(r["gt"]) - len(matches)
        ious.extend(m[0] for m in matches)
        matched_det = {m[1] for m in matches}
        mains = [r["dets"][i] for i in matched_det] + list(r["gt"])
        for i, d in enumerate(r["dets"]):
            if i in matched_det:
                continue
            if looks_like_ruby(d, mains, r["vertical"]):
                ruby_fp += 1
            else:
                other_fp += 1
        fp += len(r["dets"]) - len(matches)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    prec_x = tp / max(1, tp + other_fp)  # ruby FPs forgiven
    f1_x = 2 * prec_x * rec / max(1e-9, prec_x + rec)
    return {
        "tp": tp, "fp": fp, "fn": fn, "ruby_fp": ruby_fp, "other_fp": other_fp,
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "precision_excl_ruby": round(prec_x, 4), "f1_excl_ruby": round(f1_x, 4),
        "mean_matched_iou": round(sum(ious) / max(1, len(ious)), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="ground_truth")
    ap.add_argument("--stage", default="main", choices=["main", "refine"])
    ap.add_argument("--min-line-px", default="12",
                    help="main: single value; refine: comma list")
    ap.add_argument("--min-short-side", default="6",
                    help="main: single value; refine: comma list")
    ap.add_argument("--scale", type=int, default=1280)
    ap.add_argument("--thresh", type=float, default=0.15)
    ap.add_argument("--box-thresh", type=float, default=0.25)
    ap.add_argument("--unclip", type=float, default=1.4)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    args.min_line_px = [float(v) for v in str(args.min_line_px).split(",")]
    args.min_short_side = [int(v) for v in str(args.min_short_side).split(",")]
    if args.stage == "main":
        args.min_line_px = args.min_line_px[0]
        args.min_short_side = args.min_short_side[0]

    pages = load_gt_pages(args.gt_dir)
    n_lines = sum(len(p["lines"]) for p in pages)
    print(f"{len(pages)} pages, {n_lines} GT lines")

    det = LineDetector(model_path=resolve_lines_model("")).load()
    contexts = {k: build_contexts(pages, k) for k in ("gt-crop", "full")}
    for k, ctx in contexts.items():
        print(f"context {k}: {len(ctx.items)} detection inputs")

    rows = []
    if args.stage == "main":
        grid = []
        for scale in (960, 1280, 1600):
            det.det_long_side = scale
            forwards(contexts["gt-crop"], det)
            forwards(contexts["full"], det)
            for thresh in (0.10, 0.15, 0.20, 0.30):
                for box_thresh in (0.20, 0.25, 0.30, 0.40):
                    for unclip in (1.2, 1.4, 1.6, 1.8):
                        grid.append((scale, thresh, box_thresh, unclip))
    else:
        # refine: size gates around a fixed post-config (all post-hoc, cheap)
        s, t, bt, u = (args.scale, args.thresh, args.box_thresh, args.unclip)
        print(f"refining around scale={s} thresh={t} box={bt} unclip={u}")
        grid = [(s, t, bt, u)]
        det.det_long_side = s
        forwards(contexts["gt-crop"], det)
        forwards(contexts["full"], det)

    for scale, thresh, box_thresh, unclip in grid:
        det.det_long_side = scale
        det.thresh, det.box_thresh, det.unclip_ratio = thresh, box_thresh, unclip
        mss_list = (args.min_short_side,) if args.stage == "main" else tuple(args.min_short_side)
        for mss in mss_list:
            det.min_short_side = mss
            mlp_list = (args.min_line_px,) if args.stage == "main" else tuple(args.min_line_px)
            for mlp in mlp_list:
                row = {"scale": scale, "thresh": thresh, "box_thresh": box_thresh,
                       "unclip": unclip, "min_short_side": mss, "min_line_px": mlp,
                       "metrics": {}}
                for cname, ctx in contexts.items():
                    res = detect(ctx, det, mlp)
                    row["metrics"][cname] = evaluate(res, args.iou)
                rows.append(row)
                g = row["metrics"]["gt-crop"]
                f = row["metrics"]["full"]
                print(f"scale={scale} th={thresh} bt={box_thresh} uc={unclip} "
                      f"mss={mss} mlp={mlp} | "
                      f"gt-crop F1={g['f1']:.3f} F1x={g['f1_excl_ruby']:.3f} "
                      f"P={g['precision']:.3f} R={g['recall']:.3f} "
                      f"rubyFP={g['ruby_fp']} otherFP={g['other_fp']} IoU={g['mean_matched_iou']:.3f} | "
                      f"full F1={f['f1']:.3f} F1x={f['f1_excl_ruby']:.3f} "
                      f"P={f['precision']:.3f} R={f['recall']:.3f} "
                      f"rubyFP={f['ruby_fp']} otherFP={f['other_fp']}", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
