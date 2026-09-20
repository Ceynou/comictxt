"""Per-stage debug reporter: annotated images + values + HTML page.

Runs the real pipeline stages (region -> pad -> lines/contour/unclip/quad ->
recognition -> furigana -> paragraphs/reorder) while recording the
intermediate values at each step, then writes an ``index.html`` report with
annotated PNGs and JSON dumps.

Works with stubbed detectors (falls back to plain ``detect_pil`` when the
``*_debug`` hooks are unavailable), so it is usable in tests without weights.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from comictxt.furigana import explain_filter
from comictxt.geometry import expand_box, polygon_to_xyxy, quad_true_size
from comictxt.grouping import build_paragraph
from comictxt.io_utils import load_pil


def _round_box(b) -> list:
    return [round(float(v), 1) for v in b]


def _poly_to_list(poly) -> list:
    return [[round(float(x), 1), round(float(y), 1)] for x, y in np.asarray(poly).tolist()]


def _font(size: int = 14):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # old Pillow without size kwarg
        return ImageFont.load_default()


def _draw_box(draw: ImageDraw.ImageDraw, box, outline, width=2, label=""):
    x1, y1, x2, y2 = (float(v) for v in box)
    draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)
    if label:
        draw.text((x1 + 2, y1 + 2), label, fill=outline, font=_font())


def _draw_poly(draw: ImageDraw.ImageDraw, poly, outline, width=2, label=""):
    pts = [(float(x), float(y)) for x, y in np.asarray(poly).tolist()]
    if len(pts) >= 2:
        draw.line(pts + [pts[0]], fill=outline, width=width, joint="curve")
    if label and pts:
        draw.text((pts[0][0] + 2, pts[0][1] + 2), label, fill=outline, font=_font())


def _save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _region_debug(pipe, img: Image.Image) -> dict:
    """Return region intermediates, degrading gracefully for stub detectors."""
    debug = getattr(pipe.region, "detect_pil_debug", None)
    if debug is not None:
        try:
            info = debug(img)
            info["boxes"] = np.asarray(info["boxes"], dtype=np.float32)
            info["scores"] = np.asarray(info["scores"], dtype=np.float32)
            return info
        except Exception as e:  # noqa: BLE001 - report must not crash on backend errors
            return {"backend": "error", "error": str(e),
                    "boxes": np.zeros((0, 4), np.float32), "scores": np.zeros((0,), np.float32)}
    try:
        boxes, scores = pipe.region.detect_pil(img)
    except Exception as e:  # noqa: BLE001
        return {"backend": "error", "error": str(e),
                "boxes": np.zeros((0, 4), np.float32), "scores": np.zeros((0,), np.float32)}
    return {"backend": "plain", "boxes": np.asarray(boxes, np.float32),
            "scores": np.asarray(scores, np.float32)}


def _lines_debug(pipe, crop: Image.Image) -> dict:
    """Return line-detection intermediates for a region crop."""
    lines_obj = pipe.lines
    debug = getattr(lines_obj, "detect_bgr_debug", None)
    if debug is not None:
        try:
            arr = np.array(crop.convert("RGB"))[:, :, ::-1]
            return debug(arr)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e), "candidates": [], "kept": []}
    try:
        polys = lines_obj.detect_pil(crop)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "candidates": [], "kept": []}
    return {
        "candidates": [],
        "kept": [{"score": None, "poly": np.asarray(p, dtype=np.float32)} for p in polys],
    }


def run_debug(
    image: Union[str, Path, Image.Image],
    out_dir: Union[str, Path],
    cfg=None,
    pipe=None,
) -> Path:
    """Run the instrumented pipeline and write the HTML report.

    Returns the path to ``index.html``.
    """
    from comictxt.pipeline import ComicTxtPipeline

    img = load_pil(image)
    W, H = img.size
    if cfg is None and pipe is None:
        from comictxt.config import ComictxtConfig

        cfg = ComictxtConfig()
    if pipe is None:
        assert cfg is not None
        pipe = ComicTxtPipeline(cfg)
    else:
        cfg = pipe.cfg

    out = Path(out_dir)
    stages = out / "stages"
    crops = out / "crops"
    stages.mkdir(parents=True, exist_ok=True)
    crops.mkdir(parents=True, exist_ok=True)
    lw = max(2, max(W, H) // 500)

    sections: list[dict] = []  # {"title": str, "images": [...], "jsons": [...], "notes": [...]}

    def add_section(title, images=(), jsons=(), notes=()):
        sections.append({"title": title, "images": list(images),
                         "jsons": list(jsons), "notes": list(notes)})

    # -- stage 00: input + config -------------------------------------------
    img.save(stages / "00_input.png")
    _save_json(stages / "00_config.json", cfg.model_dump())
    add_section(
        "00 · Input + effective config",
        images=["stages/00_input.png"],
        jsons=["stages/00_config.json"],
        notes=[f"image {W}x{H}; every stage below uses these exact parameter values."],
    )

    # -- stage 01: region detection (raw/conf/NMS/containment) ---------------
    rinfo = _region_debug(pipe, img)
    boxes = np.asarray(rinfo.get("boxes", np.zeros((0, 4))), dtype=np.float32)
    scores = np.asarray(rinfo.get("scores", np.zeros((0,))), dtype=np.float32)
    rjson = {
        "backend": rinfo.get("backend"),
        "params": rinfo.get("params", {}),
        "note": rinfo.get("note", ""),
        "error": rinfo.get("error", ""),
        "n_raw": rinfo.get("n_raw"),
        "n_after_conf": rinfo.get("n_after_conf"),
        "n_after_nms": rinfo.get("n_after_nms"),
        "n_after_containment": rinfo.get("n_after_containment"),
        "n_final": int(len(boxes)),
        "boxes": [_round_box(b) for b in boxes.tolist()],
        "scores": [round(float(s), 4) for s in scores.tolist()],
    }
    _save_json(stages / "01_regions.json", rjson)
    canvas = img.copy()
    d = ImageDraw.Draw(canvas)
    pre = rinfo.get("boxes_pre_nms")
    if pre is not None and len(np.asarray(pre)) != len(boxes):
        for b in np.asarray(pre).tolist():
            _draw_box(d, b, "gray", width=max(1, lw - 1))
    for b, s in zip(boxes.tolist(), scores.tolist()):
        _draw_box(d, b, "red", width=lw, label=f"{float(s):.2f}")
    canvas.save(stages / "01_regions.png")
    notes = []
    if rinfo.get("backend") == "ultralytics":
        notes.append(rinfo.get("note", ""))
    notes.append("gray (if present) = pre-NMS proposals · red = final boxes with scores.")
    add_section("01 · Region detection (conf → NMS → containment)",
                images=["stages/01_regions.png"], jsons=["stages/01_regions.json"], notes=notes)

    # -- stage 02: padding ----------------------------------------------------
    pad_ratio = float(cfg.region.pad_ratio)
    padded_list = []
    from comictxt.geometry import expand_region_boxes

    padded_list = expand_region_boxes(
        boxes.tolist(), pad_ratio, cfg.region.pad_mode, W, H)
    _save_json(stages / "02_padding.json",
               {"pad_ratio": pad_ratio, "padded_boxes": [_round_box(b) for b in padded_list]})
    canvas = img.copy()
    d = ImageDraw.Draw(canvas)
    for b in boxes.tolist():
        _draw_box(d, b, "red", width=max(1, lw - 1))
    for b in padded_list:
        _draw_box(d, b, "orange", width=lw)
    canvas.save(stages / "02_padded.png")
    add_section("02 · Region padding", images=["stages/02_padded.png"],
                jsons=["stages/02_padding.json"],
                notes=[f"pad_ratio={pad_ratio}: red = raw region, orange = expanded crop box."])

    # -- stages 03-05: per-region lines / recognition / furigana -------------
    region_boxes = padded_list
    full_image_fallback = (
        len(boxes) == 0
        and bool(cfg.lines.run_lines_on_full_image_if_no_regions)
        and bool(cfg.lines.enable_line_stage)
    )
    if full_image_fallback:
        region_boxes = [[0, 0, W, H]]
    all_paragraphs: list[dict] = []
    final_canvas = img.copy()
    fd = ImageDraw.Draw(final_canvas)
    for ri, rb in enumerate(region_boxes):
        tag = f"full" if full_image_fallback else f"region{ri}"
        rx1, ry1, rx2, ry2 = (int(v) for v in rb)
        crop = img.crop((rx1, ry1, rx2, ry2))
        crop.save(crops / f"{tag}_crop.png")
        det_crop = pipe._cleanup_pil(crop)
        if det_crop is not crop:
            det_crop.save(crops / f"{tag}_crop_preprocessed.png")
        linfo = _lines_debug(pipe, det_crop)
        img_bgr = None
        if linfo.get("kept"):
            img_bgr = np.array(img.convert("RGB"))[:, :, ::-1]

        # draw line candidates on the crop: raw contour (gray) -> unclipped
        # polygon (yellow) -> final quad (green kept / red rejected)
        lc = crop.copy()
        ld = ImageDraw.Draw(lc)
        cand_json = []
        for ci, cand in enumerate(linfo.get("candidates", [])):
            pre_poly = cand.get("poly_pre")
            unc_poly = cand.get("poly_unclipped")
            quad = cand.get("quad")
            if pre_poly is not None:
                _draw_poly(ld, np.asarray(pre_poly) - np.array([rx1, ry1]), "gray",
                           width=max(1, lw - 1), label=f"raw{ci}")
            if unc_poly is not None:
                _draw_poly(ld, np.asarray(unc_poly) - np.array([rx1, ry1]), "orange",
                           width=max(1, lw - 1), label=f"unclip{ci}")
            if quad is not None:
                color = "green" if cand.get("kept") else "red"
                _draw_poly(ld, np.asarray(quad) - np.array([rx1, ry1]), color, width=lw,
                           label=f"{ci}")
            cand_json.append({
                "score": cand.get("score"),
                "kept": bool(cand.get("kept")),
                "reason": cand.get("reason", ""),
                "poly_pre": _poly_to_list(np.asarray(cand["poly_pre"]) - np.array([rx1, ry1]))
                if cand.get("poly_pre") is not None else None,
                "poly_unclipped": _poly_to_list(np.asarray(cand["poly_unclipped"]) - np.array([rx1, ry1]))
                if cand.get("poly_unclipped") is not None else None,
                "quad": _poly_to_list(np.asarray(cand["quad"]) - np.array([rx1, ry1]))
                if cand.get("quad") is not None else None,
            })
        kept = linfo.get("kept", [])
        if not linfo.get("candidates") and kept:
            # plain-detector fallback: draw kept quads only
            for ki, k in enumerate(kept):
                _draw_poly(ld, np.asarray(k.get("quad", k["poly"])) - np.array([rx1, ry1]),
                           "green", width=lw, label=f"{ki}")
        lc.save(stages / f"03_{tag}_lines.png")

        # recognize each kept line (min_line_px gate -> warped-quad rec)
        min_px = float(cfg.lines.min_line_px)
        line_rows = []
        line_items: list[dict] = []
        for ki, k in enumerate(kept):
            quad = np.asarray(k.get("quad", k["poly"]), dtype=np.float32)
            # kept quads from debug hook are in crop px; plain fallback in full-img px
            if linfo.get("candidates"):
                quad[:, 0] += rx1
                quad[:, 1] += ry1
            det_xyxy = polygon_to_xyxy(quad)
            gx1, gy1, gx2, gy2 = det_xyxy
            longest = max(gx2 - gx1, gy2 - gy1)
            if min_px > 0 and longest < min_px:
                line_rows.append({"quad_xyxy": _round_box([gx1, gy1, gx2, gy2]),
                                  "score": k.get("score"), "skipped": f"min_line_px {longest:.1f} < {min_px}",
                                  "text": ""})
                continue
            try:
                assert img_bgr is not None
                # Deskewed free-quad crop (same as pipeline Hayai path).
                text, xyxy = pipe._recognize_warped_quad(img_bgr, quad)
            except Exception as e:  # noqa: BLE001
                line_rows.append({"quad_xyxy": _round_box([gx1, gy1, gx2, gy2]),
                                  "score": k.get("score"), "skipped": f"recognizer error: {e}",
                                  "text": ""})
                continue
            # save the exact deskewed crop sent to the recognizer
            try:
                from comictxt.geometry import warp_quad_to_rect

                assert img_bgr is not None
                warped = warp_quad_to_rect(
                    img_bgr, np.asarray(quad, dtype=np.float32))
                Image.fromarray(warped[:, :, ::-1]).save(
                    crops / f"{tag}_line{ki}.png")
            except Exception:
                pass
            if not text:
                line_rows.append({"quad_xyxy": _round_box([gx1, gy1, gx2, gy2]),
                                  "crop_xyxy": _round_box(xyxy), "score": k.get("score"),
                                  "skipped": "empty OCR text (dropped)", "text": ""})
                continue
            line_rows.append({"quad_xyxy": _round_box([gx1, gy1, gx2, gy2]),
                              "crop_xyxy": _round_box(xyxy), "score": k.get("score"),
                              "text": text, "backend": "hayai-torch"})
            line_items.append({"text": text, "xyxy": xyxy, "det_xyxy": det_xyxy,
                               "quad": np.asarray(quad, dtype=np.float32).tolist(),
                               "sort_wh": quad_true_size(np.asarray(quad, dtype=np.float32))})
        if not line_items and bool(cfg.lines.fallback_to_region_text) and bool(cfg.lines.enable_line_stage):
            try:
                text, xyxy = pipe._recognize_box(img, float(rx1), float(ry1), float(rx2), float(ry2))
            except Exception:
                text, xyxy = "", (float(rx1), float(ry1), float(rx2), float(ry2))
            if text:
                line_rows.append({"fallback_to_region_text": True,
                                  "crop_xyxy": _round_box(xyxy), "text": text})
                line_items.append({"text": text, "xyxy": xyxy})
        _save_json(stages / f"03_{tag}_lines.json", {
            "region_xyxy": _round_box([rx1, ry1, rx2, ry2]),
            "params": linfo.get("params", {}),
            "n_contours": linfo.get("n_contours"),
            "unclip_ratio": float(cfg.lines.unclip_ratio),
            "rec_backend": "hayai-torch",
            "min_line_px": min_px,
            "candidates": cand_json,
            "recognized": line_rows,
        })
        add_section(
            f"03 · {tag}: line detection (raw → unclip → quad) + recognition",
            images=[f"stages/03_{tag}_lines.png"],
            jsons=[f"stages/03_{tag}_lines.json"],
            notes=["gray = raw contour (before any modification) · orange = unclipped "
                   "polygon · green = kept final quad · red = rejected; "
                   "line crops saved under crops/."],
        )

        # -- furigana ---------------------------------------------------------
        lc_cfg = cfg.lines
        if bool(lc_cfg.furigana_filter) and len(line_items) >= 2:
            kept_items, events = explain_filter(
                line_items,
                size_ratio=lc_cfg.furigana_size_ratio,
                proximity_ratio=lc_cfg.furigana_proximity_ratio,
                overlap_ratio=lc_cfg.furigana_overlap_ratio,
                max_thickness_px=lc_cfg.furigana_max_thickness_px,
                length_ratio=lc_cfg.furigana_length_ratio,
                char_size_ratio=lc_cfg.furigana_char_size_ratio,
                proximity_min=lc_cfg.furigana_proximity_min,
                proximity_max=lc_cfg.furigana_proximity_max,
                max_chars=lc_cfg.furigana_max_chars,
                box_pad=lc_cfg.box_pad,
            )
        else:
            kept_items, events = list(line_items), [
                {"index": i, "text": ln.get("text", ""), "dropped": False,
                 "reason": "filter disabled or <2 lines", "main_index": None}
                for i, ln in enumerate(line_items)
            ]
        _save_json(stages / f"04_{tag}_furigana.json", {
            "params": {"furigana_filter": bool(lc_cfg.furigana_filter),
                       "furigana_size_ratio": lc_cfg.furigana_size_ratio,
                       "furigana_proximity_ratio": lc_cfg.furigana_proximity_ratio,
                       "furigana_overlap_ratio": lc_cfg.furigana_overlap_ratio,
                       "furigana_max_thickness_px": lc_cfg.furigana_max_thickness_px,
                       "furigana_length_ratio": lc_cfg.furigana_length_ratio,
                       "furigana_char_size_ratio": lc_cfg.furigana_char_size_ratio,
                       "furigana_proximity_min": lc_cfg.furigana_proximity_min,
                       "furigana_proximity_max": lc_cfg.furigana_proximity_max,
                       "furigana_max_chars": lc_cfg.furigana_max_chars},
            "events": events,
        })
        fc = crop.copy()
        fdraw = ImageDraw.Draw(fc)
        for ev in events:
            i = int(ev["index"])
            if i < len(line_items):
                b = line_items[i]["xyxy"]
                local = [b[0] - rx1, b[1] - ry1, b[2] - rx1, b[3] - ry1]
                _draw_box(fdraw, local, "gray" if ev["dropped"] else "green",
                          width=lw, label=("DROP " if ev["dropped"] else "") + f"{i}")
        fc.save(stages / f"04_{tag}_furigana.png")
        add_section(
            f"04 · {tag}: furigana filter",
            images=[f"stages/04_{tag}_furigana.png"],
            jsons=[f"stages/04_{tag}_furigana.json"],
            notes=["green = kept · gray = dropped as ruby (see reasons in JSON)."],
        )

        para = build_paragraph(
            kept_items, W, H,
            None if full_image_fallback else (float(rx1), float(ry1), float(rx2), float(ry2)),
            cfg.pipeline.layout_overlap_ratio, cfg.pipeline.layout_gap_ratio,
        )
        if para is not None:
            all_paragraphs.append(para)

    # -- stage 05: paragraphs + reorder --------------------------------------
    pre_reorder = json.loads(json.dumps(all_paragraphs, ensure_ascii=False))
    final_paras = pipe._finalize_paragraphs(all_paragraphs)
    result = {"image_properties": {"width": W, "height": H},
              "engine_capabilities": dict(pipe.ENGINE_CAPABILITIES)
              if hasattr(pipe, "ENGINE_CAPABILITIES") else {},
              "paragraphs": final_paras}
    try:
        from comictxt.pipeline import ENGINE_CAPABILITIES as _EC

        result["engine_capabilities"] = dict(_EC)
    except Exception:
        pass
    _save_json(stages / "05_paragraphs.json", {
        "reorder_blocks": bool(cfg.pipeline.reorder_blocks),
        "reorder_lines": bool(cfg.pipeline.reorder_lines),
        "n_before": len(pre_reorder), "n_after": len(final_paras),
        "before_reorder": pre_reorder, "paragraphs": final_paras,
    })
    _save_json(out / "result.json", result)
    for pi, para in enumerate(final_paras):
        bb = para.get("bounding_box", {})
        cx, cy = float(bb.get("center_x", 0)) * W, float(bb.get("center_y", 0)) * H
        w, h = float(bb.get("width", 0)) * W, float(bb.get("height", 0)) * H
        _draw_box(fd, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], "blue", width=lw,
                  label=f"P{pi} {para.get('writing_direction', '')}")
        for ln in para.get("lines", []):
            lb = ln.get("bounding_box", {})
            lx, ly = float(lb.get("center_x", 0)) * W, float(lb.get("center_y", 0)) * H
            lw_, lh_ = float(lb.get("width", 0)) * W, float(lb.get("height", 0)) * H
            _draw_box(fd, [lx - lw_ / 2, ly - lh_ / 2, lx + lw_ / 2, ly + lh_ / 2],
                      "green", width=max(1, lw - 1))
    final_canvas.save(stages / "05_final.png")
    add_section("05 · Final paragraphs (blue) + lines (green) + reading order",
                images=["stages/05_final.png"],
                jsons=["stages/05_paragraphs.json", "result.json"],
                notes=["result.json is the exact owocr-compatible output."])

    # -- index.html ------------------------------------------------------------
    parts = ["<html><head><meta charset='utf-8'>",
             "<style>body{font-family:sans-serif;max-width:1100px;margin:auto;padding:16px}"
             "img{max-width:100%;border:1px solid #ccc}pre{background:#f4f4f4;padding:8px;"
             "overflow:auto;max-height:400px}h2{border-bottom:2px solid #333}</style>",
             "</head><body><h1>comictxt debug report</h1>"]
    for sec in sections:
        parts.append(f"<h2>{html.escape(sec['title'])}</h2>")
        for n in sec["notes"]:
            parts.append(f"<p>{html.escape(n)}</p>")
        for im in sec["images"]:
            parts.append(f"<img src='{html.escape(im)}'><p><code>{html.escape(im)}</code></p>")
        for js in sec["jsons"]:
            try:
                txt = (out / js).read_text(encoding="utf-8")
            except FileNotFoundError:
                txt = "(missing)"
            parts.append(f"<h3><code>{html.escape(js)}</code></h3><pre>{html.escape(txt)}</pre>")
    parts.append("</body></html>")
    index = out / "index.html"
    index.write_text("\n".join(parts), encoding="utf-8")
    return index
