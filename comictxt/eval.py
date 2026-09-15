"""Eval against mokuro-format ground truth (ground_truth/*.json + *.webp).

Metrics per page: block detection precision/recall (IoU>=0.5 greedy match),
writing-direction accuracy on matched blocks, line-text CER on matched
blocks (concatenated lines, NFKC + CJK space-strip normalization).
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"[\r\n\t]+", " ", text)
    cjk = r"[\u4e00-\u9fff\u3040-\u30ff\u3400-\u4dbf\U00020000-\U0002a6df\ac00-\ud7af]"
    text = re.sub(f"({cjk})\\s+({cjk})", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    ref, hyp = normalize_text(ref), normalize_text(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / max(1, len(ref))


def box_iou(a: list, b: list) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def owocr_to_blocks(owocr: dict) -> list[dict]:
    """Mirror of neokuro converter: owocr paragraphs -> mokuro-style blocks."""
    W = owocr["image_properties"]["width"]
    H = owocr["image_properties"]["height"]
    blocks = []
    for para in owocr.get("paragraphs", []):
        vertical = para.get("writing_direction") == "TOP_TO_BOTTOM"
        texts, boxes = [], []
        for line in para.get("lines", []):
            bb = line["bounding_box"]
            cx, cy = bb["center_x"] * W, bb["center_y"] * H
            w, h = bb["width"] * W, bb["height"] * H
            texts.append(line["text"])
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        if not boxes:
            continue
        blocks.append(
            {
                "box": [
                    min(b[0] for b in boxes),
                    min(b[1] for b in boxes),
                    max(b[2] for b in boxes),
                    max(b[3] for b in boxes),
                ],
                "vertical": vertical,
                "lines": texts,
            }
        )
    return blocks


def match_blocks(pred: list[dict], gt: list[dict], iou_thresh: float = 0.5):
    """Greedy IoU matching. Returns list of (pred_idx, gt_idx)."""
    cands = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            iou = box_iou(p["box"], g["box"])
            if iou >= iou_thresh:
                cands.append((iou, i, j))
    cands.sort(reverse=True)
    used_p, used_g, matches = set(), set(), []
    for _, i, j in cands:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            matches.append((i, j))
    return matches


def eval_page(gt_path: Path, pred_blocks: list[dict], iou_thresh: float = 0.5) -> dict:
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    gt_blocks = gt["blocks"]
    matches = match_blocks(pred_blocks, gt_blocks, iou_thresh)
    tp = len(matches)
    precision = tp / max(1, len(pred_blocks))
    recall = tp / max(1, len(gt_blocks))
    vert_ok = sum(1 for i, j in matches if pred_blocks[i]["vertical"] == gt_blocks[j]["vertical"])
    cers = []
    for i, j in matches:
        ref = " ".join(gt_blocks[j]["lines"])
        hyp = " ".join(pred_blocks[i]["lines"])
        cers.append(cer(ref, hyp))
    return {
        "page": gt_path.stem,
        "gt_blocks": len(gt_blocks),
        "pred_blocks": len(pred_blocks),
        "matched": tp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "vertical_acc": round(vert_ok / max(1, tp), 4),
        "mean_cer": round(sum(cers) / max(1, len(cers)), 4),
    }


def summarize(page_results: list[dict]) -> dict:
    tp = sum(p["matched"] for p in page_results)
    np_ = sum(p["pred_blocks"] for p in page_results)
    ng = sum(p["gt_blocks"] for p in page_results)
    vo = sum(p["vertical_acc"] * p["matched"] for p in page_results)
    ce = sum(p["mean_cer"] * p["matched"] for p in page_results)
    return {
        "pages": page_results,
        "total": {
            "precision": round(tp / max(1, np_), 4),
            "recall": round(tp / max(1, ng), 4),
            "vertical_acc": round(vo / max(1, tp), 4),
            "mean_cer": round(ce / max(1, tp), 4),
            "matched": tp,
            "pred_blocks": np_,
            "gt_blocks": ng,
        },
    }


IMAGE_SUFFIXES = (".webp", ".png", ".jpg", ".jpeg", ".jxl")


def find_image(gt_path: Path) -> Path | None:
    for suf in IMAGE_SUFFIXES:
        cand = gt_path.with_suffix(suf)
        if cand.exists():
            return cand
    return None


def run_eval(gt_dir: Path, pipeline, iou_thresh: float = 0.5) -> dict:
    results = []
    for gt_path in sorted(gt_dir.glob("*.json")):
        img_path = find_image(gt_path)
        if img_path is None:
            continue
        owocr = pipeline.process_image(img_path)
        results.append(eval_page(gt_path, owocr_to_blocks(owocr), iou_thresh))
    return summarize(results)
