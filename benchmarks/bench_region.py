"""Region-detection benchmark (AnimeText YOLO) against GT blocks.

Caches the raw ONNX output per (size, imgsz) at conf=0.01/no-NMS/keep, then
sweeps conf x nms x containment post-hoc (exactly the backend's own post
steps), evaluating block P/R/F1 (IoU>=0.5) like comictxt eval. Top configs
are re-validated on the ultralytics .pt backend (pipeline default).

Usage:
  python benchmarks/bench_region.py [--nms] [--top-ultralytics N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gt import load_gt_pages  # noqa: E402

from comictxt.config import resolve_region_onnx, resolve_region_pt  # noqa: E402
from comictxt.eval import match_blocks  # noqa: E402
from comictxt.geometry import (  # noqa: E402
    containment_suppress,
    merge_contained_boxes,
    nms,
)
from comictxt.region_yolo import OnnxYoloBackend, UltralyticsYoloBackend  # noqa: E402


def postprocess(boxes: np.ndarray, scores: np.ndarray, conf: float, nms_on: bool,
                action: str, cthresh: float, iou: float, max_det: int):
    """Mirror of OnnxYoloBackend post-forward steps."""
    keep = scores >= conf
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) == 0:
        return boxes, scores
    order = np.argsort(-scores)[: max_det * 4]
    boxes, scores = boxes[order], scores[order]
    if nms_on:
        kept = nms(boxes, scores, iou)[:max_det]
        boxes, scores = boxes[kept], scores[kept]
    else:
        boxes, scores = boxes[:max_det], scores[:max_det]
    if len(boxes) > 1 and cthresh > 0:
        if action == "merge":
            boxes, scores = merge_contained_boxes(boxes, scores, cthresh)
        elif action == "drop":
            kept = containment_suppress(boxes, scores, cthresh)
            boxes, scores = boxes[kept], scores[kept]
    return boxes, scores


def eval_boxes(pages, page_boxes) -> dict:
    tp = fp = fn = 0
    for p, boxes in zip(pages, page_boxes):
        pred = [{"box": b.tolist()} for b in np.asarray(boxes)]
        gt = [{"box": g["box"]} for g in p["blocks"]]
        matches = match_blocks(pred, gt, 0.5)
        tp += len(matches)
        fp += len(pred) - len(matches)
        fn += len(gt) - len(matches)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(1e-9, prec + rec), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="ground_truth")
    ap.add_argument("--sizes", default="n,s,m,l,x")
    ap.add_argument("--confs", default="0.15,0.20,0.25,0.30,0.40")
    ap.add_argument("--imgszs", default="640,960,1280")
    ap.add_argument("--contains", default="merge@0.8,drop@0.8,keep@0")
    ap.add_argument("--nms", action="store_true", help="also try IoU-NMS on")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--top-ultralytics", type=int, default=3,
                    help="re-validate top N distinct configs with .pt (0=off)")
    ap.add_argument("--out", default="benchmarks/results/region_sweep.json")
    args = ap.parse_args()

    pages = load_gt_pages(args.gt_dir)
    n_blocks = sum(len(p["blocks"]) for p in pages)
    print(f"{len(pages)} pages, {n_blocks} GT blocks")

    sizes = args.sizes.split(",")
    confs = [float(c) for c in args.confs.split(",")]
    imgszs = [int(s) for s in args.imgszs.split(",")]
    contains = []
    for c in args.contains.split(","):
        action, thresh = c.split("@")
        contains.append((action, float(thresh)))
    nms_modes = [False, True] if args.nms else [False]

    rows = []
    for size in sizes:
        model_path = resolve_region_onnx(size, "")
        for imgsz in imgszs:
            be = OnnxYoloBackend(
                model_path=model_path, imgsz=(imgsz, imgsz),
                conf=0.01, iou=args.iou, nms=False, max_det=300,
                contain_thresh=0.0, contain_action="keep",
            ).load()
            raws = []
            for p in pages:
                info = be.detect_pil_debug(p["img"])
                raws.append((info["boxes_pre_nms"].astype(np.float64),
                             info["scores_pre_nms"].astype(np.float64)))
            for nms_on in nms_modes:
                for action, cthresh in contains:
                    for conf in confs:
                        page_boxes = []
                        for boxes, scores in raws:
                            b, s = postprocess(boxes.copy(), scores.copy(), conf,
                                               nms_on, action, cthresh,
                                               args.iou, 300)
                            page_boxes.append(b)
                        m = eval_boxes(pages, page_boxes)
                        rows.append({"size": size, "imgsz": imgsz, "conf": conf,
                                     "nms": nms_on,
                                     "contain": f"{action}@{cthresh}",
                                     "metrics": m})
                        print(f"{size} imgsz={imgsz} conf={conf:.2f} nms={int(nms_on)} "
                              f"contain={action}@{cthresh} | F1={m['f1']:.3f} "
                              f"P={m['precision']:.3f} R={m['recall']:.3f} "
                              f"tp={m['tp']} fp={m['fp']} fn={m['fn']}", flush=True)

    rows.sort(key=lambda r: -r["metrics"]["f1"])
    print("\n== top 10 (onnx) ==")
    for r in rows[:10]:
        print(f"{r['size']:>2} imgsz={r['imgsz']} conf={r['conf']:.2f} "
              f"nms={int(r['nms'])} contain={r['contain']:<9} "
              f"F1={r['metrics']['f1']:.3f} P={r['metrics']['precision']:.3f} "
              f"R={r['metrics']['recall']:.3f}")

    if args.top_ultralytics:
        print("\n== ultralytics parity (contain=merge@0.8, iou=0.7) ==")
        seen = set()
        checked = 0
        for r in rows:
            key = (r["size"], r["imgsz"], r["conf"])
            if key in seen:
                continue
            seen.add(key)
            be = UltralyticsYoloBackend(
                model_path=resolve_region_pt(r["size"], ""),
                imgsz=(r["imgsz"], r["imgsz"]), conf=r["conf"], iou=args.iou,
                max_det=300, contain_thresh=0.80, contain_action="merge",
            ).load()
            page_boxes = []
            for p in pages:
                b, _ = be.detect_pil(p["img"])
                page_boxes.append(b)
            m = eval_boxes(pages, page_boxes)
            r["metrics_ultralytics"] = m
            print(f"ULTRA {r['size']:>2} imgsz={r['imgsz']} conf={r['conf']:.2f} | "
                  f"F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"tp={m['tp']} fp={m['fp']} fn={m['fn']}")
            checked += 1
            if checked >= args.top_ultralytics:
                break

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
