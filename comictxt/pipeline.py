"""End-to-end pipeline: region -> (lines) -> recognition -> owocr JSON."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from comictxt.config import ComictxtConfig, resolve_workers
from comictxt.furigana import filter_furigana
from comictxt.geometry import (
    clip_box,
    expand_box,
    ink_size,
    polygon_to_xyxy,
    quad_true_size,
    warp_quad_to_rect,
)
from comictxt.grouping import _cluster_adjacent, apply_reading_order, build_paragraph
from comictxt.io_utils import load_pil
from comictxt.lines_ppocr import LineDetector
from comictxt.preprocess import cleanup as cleanup_image
from comictxt.rec_hayai import HayaiRecognizer
from comictxt.region_yolo import RegionDetector

log = logging.getLogger(__name__)

ENGINE_CAPABILITIES = {
    "symbols": False,
    "symbol_bounding_boxes": False,
    "words": True,
    "word_bounding_boxes": True,
    "lines": True,
    "line_bounding_boxes": True,
    "paragraphs": True,
    "paragraph_bounding_boxes": True,
}


def _is_ruby_of(candidate: dict, main: dict, cfg) -> bool:
    """True if ``candidate`` reads as ruby over ``main`` under either layout.

    The orphan sweep cannot trust a local orientation vote (a single wide
    junk neighbor flips it), so both the vertical (ruby right of main) and
    horizontal (ruby above main) rule sets are tried.
    """
    from comictxt.furigana import is_furigana_pair

    lc = cfg.lines
    for vertical in (True, False):
        if is_furigana_pair(
            candidate, main,
            is_vertical=vertical,
            size_ratio=lc.furigana_size_ratio,
            proximity_ratio=lc.furigana_proximity_ratio,
            overlap_ratio=lc.furigana_overlap_ratio,
            max_thickness_px=lc.furigana_max_thickness_px,
            length_ratio=lc.furigana_length_ratio,
            char_size_ratio=lc.furigana_char_size_ratio,
            proximity_min=lc.furigana_proximity_min,
            proximity_max=lc.furigana_proximity_max,
            max_chars=lc.furigana_max_chars,
            box_pad=lc.box_pad,
        ):
            return True
    return False


class ComicTxtPipeline:
    def __init__(self, config: ComictxtConfig | None = None, allow_inner_parallel: bool = True) -> None:
        self.cfg = config or ComictxtConfig()
        # False when an outer pool already fans out (batch CLI threads mode):
        # avoids threads^2 oversubscription and nested-executor deadlocks.
        self.allow_inner_parallel = allow_inner_parallel
        self._region: RegionDetector | None = None
        self._lines: LineDetector | None = None
        self._rec = None

    # -- lazy components ---------------------------------------------------
    @property
    def region(self) -> RegionDetector:
        if self._region is None:
            rc = self.cfg.region
            self._region = RegionDetector(
                backend=rc.backend,
                model_size=rc.model_size,
                onnx_path=rc.onnx_path,
                pt_path=rc.pt_path,
                imgsz=rc.resolved_imgsz(),
                conf=rc.resolved_conf(),
                iou=rc.iou,
                nms=rc.nms,
                max_det=rc.max_det,
                providers=list(rc.providers),
                device=rc.device,
                contain_thresh=rc.contain_thresh,
                contain_action=rc.contain_action,
            ).load()
        return self._region

    @property
    def lines(self) -> LineDetector:
        if self._lines is None:
            lc = self.cfg.lines
            self._lines = LineDetector(
                model_path=lc.resolved_model(),
                det_long_side=lc.det_long_side,
                det_min_side=lc.det_min_side,
                det_margin=lc.det_margin,
                thresh=lc.thresh,
                box_thresh=lc.box_thresh,
                unclip_ratio=lc.unclip_ratio,
                min_short_side=lc.min_short_side,
                box_pad=lc.box_pad,
                max_candidates=lc.max_candidates,
                providers=list(lc.providers),
            ).load()
        return self._lines

    @property
    def rec(self):  # Union[HayaiRecognizer, TorchHayaiRecognizer, PpocrRecognizer]
        if self._rec is None:
            rc = self.cfg.rec
            if rc.backend == "torch":
                from comictxt.rec_hayai_torch import TorchHayaiRecognizer, resolve_rec_processor

                from comictxt.config import is_offline

                self._rec = TorchHayaiRecognizer(
                    model_path=rc.resolved_torch_model(),
                    processor_path=resolve_rec_processor(rc.torch_processor),
                    device=rc.device,
                    dtype=rc.dtype,
                    max_new_tokens=rc.max_new_tokens,
                    max_num_patches=rc.max_num_patches,
                    offline=is_offline(self.cfg),
                ).load()
            elif rc.backend == "ppocr":
                from comictxt.rec_ppocr import PpocrRecognizer

                pc = self.cfg.preprocess
                self._rec = PpocrRecognizer(
                    model_path=rc.resolved_ppocr_model(),
                    dict_path=rc.resolved_ppocr_dict(),
                    providers=list(rc.providers),
                    trim=rc.ppocr_trim,
                    preprocess={
                        "enable": pc.enable,
                        "black_point": pc.black_point,
                        "white_point": pc.white_point,
                        "sharpen": pc.sharpen,
                    },
                ).load()
            else:
                self._rec = HayaiRecognizer(
                    onnx_dir=rc.resolved_onnx_dir(),
                    precision=rc.precision,
                    providers=list(rc.providers),
                    max_new_tokens=rc.max_new_tokens,
                ).load()
        return self._rec

    def warmup(self) -> None:
        self.region.ensure_loaded()
        if self.cfg.lines.enable_line_stage:
            self.lines.ensure_loaded()
        self.rec.ensure_loaded()

    # -- cleanup / crop helpers ----------------------------------------------
    def _cleanup_pil(self, crop: Image.Image) -> Image.Image:
        """Optional leveling/sharpening on a PIL crop (passthrough if off)."""
        pc = self.cfg.preprocess
        if not pc.enable:
            return crop
        arr = np.array(crop.convert("RGB"))[:, :, ::-1]  # RGB -> BGR
        out = cleanup_image(
            arr, enable=True, black_point=pc.black_point,
            white_point=pc.white_point, sharpen=pc.sharpen,
        )
        return Image.fromarray(out[:, :, ::-1])  # BGR -> RGB

    def _recognize_box(
        self, img: Image.Image, x1: float, y1: float, x2: float, y2: float
    ) -> tuple[str, tuple[float, float, float, float]]:
        W, H = img.size
        pad = int(self.cfg.rec.line_pad)
        ix1, iy1, ix2, iy2 = clip_box(x1 - pad, y1 - pad, x2 + pad, y2 + pad, W, H)
        # enforce minimum size by expanding around center
        min_s = int(self.cfg.rec.min_crop_size)
        if ix2 - ix1 < min_s or iy2 - iy1 < min_s:
            cx, cy = (ix1 + ix2) / 2.0, (iy1 + iy2) / 2.0
            hw = max(min_s, ix2 - ix1) / 2.0
            hh = max(min_s, iy2 - iy1) / 2.0
            ix1, iy1, ix2, iy2 = clip_box(cx - hw, cy - hh, cx + hw, cy + hh, W, H)
        if ix2 <= ix1 or iy2 <= iy1:
            return "", (x1, y1, x2, y2)
        crop = img.crop((ix1, iy1, ix2, iy2))
        crop = self._cleanup_pil(crop)
        blank_thresh = float(self.cfg.rec.blank_std_thresh)
        if blank_thresh > 0:
            gray = np.asarray(crop.convert("L"), dtype=np.float32)
            if float(gray.std()) < blank_thresh:
                return "", (float(ix1), float(iy1), float(ix2), float(iy2))
        # NOTE: no 90-degree rotation for Hayai (2D-RoPE VLM handles vertical natively;
        # rotation is a PP-OCR CTC convention and hurts Hayai accuracy).
        text = self.rec.ocr_pil(crop)
        return text.strip(), (float(ix1), float(iy1), float(ix2), float(iy2))

    def _finalize_paragraphs(self, paragraphs: list[dict]) -> list[dict]:
        """Drop empty paragraphs and apply reading-order reorder."""
        kept = [
            p
            for p in paragraphs
            if p is not None
            and p.get("lines")
            and any(str(ln.get("text", "")).strip() for ln in p["lines"])
        ]
        if not kept:
            return []
        pc = self.cfg.pipeline
        if pc.reorder_blocks or pc.reorder_lines:
            kept = apply_reading_order(
                kept,
                reorder_blocks=bool(pc.reorder_blocks),
                reorder_lines=bool(pc.reorder_lines),
            )
        return kept

    def _maybe_defurigana(self, line_items: list[dict]) -> list[dict]:
        lc = self.cfg.lines
        if not lc.furigana_filter or len(line_items) < 2:
            return line_items
        return filter_furigana(
            line_items,
            size_ratio=lc.furigana_size_ratio,
            proximity_ratio=lc.furigana_proximity_ratio,
            overlap_ratio=lc.furigana_overlap_ratio,
            max_thickness_px=lc.furigana_max_thickness_px,
            length_ratio=lc.furigana_length_ratio,
            char_size_ratio=lc.furigana_char_size_ratio,
            proximity_min=lc.furigana_proximity_min,
            proximity_max=lc.furigana_proximity_max,
            max_chars=lc.furigana_max_chars,
            is_vertical=None,  # auto-vote per region (Space orientation vote)
            box_pad=lc.box_pad,  # undo detector quad inflation in thickness ratios
        )

    def _recognize_quad(
        self, img_bgr: np.ndarray, quad: np.ndarray
    ) -> tuple[str, tuple[float, float, float, float]]:
        """PP-OCR path: perspective-warp the detector quad and decode.

        Returns (text, updated quad xyxy) — the box comes back from the warp
        corners, exactly like the Space app.
        """
        text, new_quad = self.rec.ocr_quad(img_bgr, np.asarray(quad, dtype=np.float32))
        # NOTE: no blank-std gate here on purpose — CTC emits "" on blank
        # crops (dropped below), unlike VLMs which hallucinate.
        return text.strip(), polygon_to_xyxy(np.asarray(new_quad, dtype=np.float32))

    def _recognize_warped_quad(
        self, img_bgr: np.ndarray, quad: np.ndarray
    ) -> tuple[str, tuple[float, float, float, float]]:
        """Hayai (onnx/torch) path: deskew the rotated detector quad.

        The old code fed ``polygon_to_xyxy(quad)`` (axis-aligned bbox) to
        ``_recognize_box`` — for rotated quads that bbox includes background
        and neighbor characters, causing repeated/hallucinated text. This
        warps the free quad to a tight upright rectangle instead (vertical
        stays vertical: no rotate-if-tall, unlike the PP-OCR CTC path).
        Returns (text, tight quad xyxy); padding is recognition context only
        and is not folded into the output box.
        """
        import cv2

        q = np.asarray(quad, dtype=np.float32)
        det_xyxy = polygon_to_xyxy(q)
        warped = warp_quad_to_rect(img_bgr, q)
        if warped.size == 0 or warped.shape[0] < 2 or warped.shape[1] < 2:
            return "", det_xyxy
        pad = int(self.cfg.rec.line_pad)
        if pad > 0:
            warped = cv2.copyMakeBorder(
                warped, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        min_s = int(self.cfg.rec.min_crop_size)
        if min_s > 0 and (warped.shape[0] < min_s or warped.shape[1] < min_s):
            scale = max(min_s / max(1, warped.shape[0]),
                        min_s / max(1, warped.shape[1]))
            # cap runaway upscales on degenerate 1-2px warps
            scale = min(scale, 8.0)
            warped = cv2.resize(
                warped, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        # BGR -> RGB PIL for cleanup + recognizer
        crop = Image.fromarray(warped[:, :, ::-1])
        crop = self._cleanup_pil(crop)
        blank_thresh = float(self.cfg.rec.blank_std_thresh)
        if blank_thresh > 0:
            gray = np.asarray(crop.convert("L"), dtype=np.float32)
            if float(gray.std()) < blank_thresh:
                return "", det_xyxy
        # NOTE: no 90-degree rotation for Hayai (2D-RoPE VLM handles vertical
        # natively; rotation is a PP-OCR CTC convention and hurts accuracy).
        text = self.rec.ocr_pil(crop)
        return text.strip(), det_xyxy

    @staticmethod
    def _junk_single_char(text: str) -> bool:
        """True for one-alphanumeric-letter texts (artwork junk fragments).

        Punctuation-only lines (ー, !!) are legitimate manga text and pass.
        """
        stripped = (text or "").strip()
        if len(stripped) != 1:
            return False
        return stripped[0].isalnum()

    def _expand_region_boxes(
        self, boxes: list[tuple], W: int, H: int
    ) -> list[tuple]:
        """Expand region boxes per ``region.pad_mode`` (auto/uniform/proportional/max).

        ``auto``: uniform for isolated regions (tight long-line crops gain
        side context), proportional when the widened box would reach into
        another region (adjacent bubbles/columns stay separate). See
        ``geometry.expand_region_boxes``.
        """
        from comictxt.geometry import expand_region_boxes

        rc = self.cfg.region
        return expand_region_boxes(boxes, rc.pad_ratio, rc.pad_mode, W, H)

    # -- per-region work (line-det + rec; thread-safe, order restored by caller)
    def _process_region(self, img: Image.Image, box: tuple) -> tuple[tuple, list[dict]]:
        """``box`` is the already-expanded crop box (see _expand_region_boxes)."""
        cfg = self.cfg
        rx1, ry1, rx2, ry2 = (float(v) for v in box)
        if rx2 <= rx1 or ry2 <= ry1:
            return (rx1, ry1, rx2, ry2), []
        region_crop = self._cleanup_pil(img.crop((rx1, ry1, rx2, ry2)))
        line_items: list[dict] = []
        min_px = float(cfg.lines.min_line_px)
        min_chars = int(cfg.lines.min_line_chars)
        use_ppocr = cfg.rec.backend == "ppocr"
        img_bgr = None
        if cfg.lines.enable_line_stage:
            quads = self.lines.detect_pil(region_crop)
            if quads:
                img_bgr = np.array(img.convert("RGB"))[:, :, ::-1]
            for quad in quads:
                q = np.asarray(quad, dtype=np.float32)
                q[:, 0] += rx1
                q[:, 1] += ry1
                det_xyxy = polygon_to_xyxy(q)  # tight box for furigana rules
                sort_wh = quad_true_size(q)  # deskewed size for reading order
                ink_wh = ink_size(img_bgr, q) if img_bgr is not None else None
                if use_ppocr:
                    assert img_bgr is not None
                    text, xyxy = self._recognize_quad(img_bgr, q)
                    if not text:
                        continue
                    if min_chars > 1 and self._junk_single_char(text):
                        continue
                    line_items.append(
                        {"text": text, "xyxy": xyxy, "det_xyxy": det_xyxy,
                         "quad": q.tolist(), "sort_wh": sort_wh, "ink_wh": ink_wh})
                    continue
                px1, py1, px2, py2 = det_xyxy
                if min_px > 0 and max(px2 - px1, py2 - py1) < min_px:
                    continue
                assert img_bgr is not None
                text, xyxy = self._recognize_warped_quad(img_bgr, q)
                if not text:
                    continue
                if min_chars > 1 and self._junk_single_char(text):
                    continue
                line_items.append(
                    {"text": text, "xyxy": xyxy, "det_xyxy": det_xyxy,
                     "quad": q.tolist(), "sort_wh": sort_wh, "ink_wh": ink_wh})
            if not line_items and cfg.lines.fallback_to_region_text:
                text, xyxy = self._recognize_box(img, float(rx1), float(ry1), float(rx2), float(ry2))
                if text:
                    line_items.append({"text": text, "xyxy": xyxy})
        else:
            text, xyxy = self._recognize_box(img, float(rx1), float(ry1), float(rx2), float(ry2))
            if text:
                line_items.append({"text": text, "xyxy": xyxy})
        return (float(rx1), float(ry1), float(rx2), float(ry2)), line_items

    # -- orphan recovery (lines missed by the region flow) --------------------
    def _orphan_paragraphs(
        self, img: Image.Image, existing_line_items: list[dict] | None = None
    ) -> list[dict]:
        """Full-page line sweep for text the per-region flow missed.

        The line detector runs once over the whole page. A quad is a
        duplicate — not an orphan — when it overlaps an already-recognized
        line's detection box (IoU or center-inside). Surviving quads are
        recognized (orphans on raw artwork must carry at least
        ``orphan_min_chars`` alphanumeric characters), clustered into
        paragraphs (adjacency rules mirror ``infer_orientation`` pairs),
        and furigana-checked against nearby existing lines (under both
        orientations — junk neighbors can flip a local vote) so ruby next
        to a recognized main line is not re-adopted. Existing lines keep
        their detector boxes here: mixed box provenance (rec-trimmed vs
        det) breaks the thickness ratios.
        """
        cfg = self.cfg
        if not cfg.lines.enable_line_stage or not cfg.lines.orphan_sweep:
            return []
        if not existing_line_items and cfg.lines.run_lines_on_full_image_if_no_regions:
            return []  # the no-region branch already ran full-image lines
        W, H = img.size
        quads = self.lines.detect_pil(self._cleanup_pil(img))
        if not quads:
            return []
        min_px = float(cfg.lines.min_line_px)
        min_chars = int(cfg.lines.orphan_min_chars)
        use_ppocr = cfg.rec.backend == "ppocr"
        img_bgr = None

        def _iou(a, b) -> float:
            ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
            ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0:
                return 0.0
            ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            return inter / ua if ua > 0 else 0.0

        existing = list(existing_line_items or [])

        def _box(it: dict) -> list[float]:
            return it.get("det_xyxy", it.get("xyxy"))

        items: list[dict] = []
        for quad in quads:
            q = np.asarray(quad, dtype=np.float32)
            det_xyxy = polygon_to_xyxy(q)
            cx = (det_xyxy[0] + det_xyxy[2]) / 2.0
            cy = (det_xyxy[1] + det_xyxy[3]) / 2.0
            if any(
                _iou(det_xyxy, eb) > 0.45 or (eb[0] <= cx <= eb[2] and eb[1] <= cy <= eb[3])
                for eb in (_box(e) for e in existing)
            ):
                continue  # duplicate of an already-recognized line
            sort_wh = quad_true_size(q)
            if not use_ppocr and min_px > 0:
                px1, py1, px2, py2 = det_xyxy
                if max(px2 - px1, py2 - py1) < min_px:
                    continue
            if img_bgr is None:
                img_bgr = np.array(img.convert("RGB"))[:, :, ::-1]
            if use_ppocr:
                text, xyxy = self._recognize_quad(img_bgr, q)
            else:
                text, xyxy = self._recognize_warped_quad(img_bgr, q)
            if not text:
                continue
            if min_chars > 0 and sum(1 for ch in text if ch.isalnum()) < min_chars:
                continue  # one-glyph junk from artwork, not text
            items.append(
                {"text": text, "xyxy": xyxy, "det_xyxy": det_xyxy,
                 "quad": q.tolist(), "sort_wh": sort_wh,
                 "ink_wh": ink_size(img_bgr, q)})
        if not items:
            return []
        clusters = _cluster_adjacent(
            [_box(it) for it in items],
            cfg.pipeline.layout_overlap_ratio, cfg.pipeline.layout_gap_ratio,
        )
        paragraphs: list[dict] = []
        for idxs in clusters:
            cluster_items = [items[i] for i in idxs]
            if cfg.lines.furigana_filter and existing:
                # Ruby may sit beside an already-recognized main line: drop
                # cluster lines that are ruby of a nearby existing line.
                # Nearby = within 4x the cluster extent around it.
                cx1 = min(_box(c)[0] for c in cluster_items)
                cy1 = min(_box(c)[1] for c in cluster_items)
                cx2 = max(_box(c)[2] for c in cluster_items)
                cy2 = max(_box(c)[3] for c in cluster_items)
                margin = 4.0 * max(cx2 - cx1, cy2 - cy1, 32.0)
                nearby = [
                    ex for ex in existing
                    if cx1 - margin <= (_box(ex)[0] + _box(ex)[2]) / 2 <= cx2 + margin
                    and cy1 - margin <= (_box(ex)[1] + _box(ex)[3]) / 2 <= cy2 + margin
                ]
                survivors = [
                    c for c in cluster_items
                    if not any(
                        _is_ruby_of(c, ex, cfg) for ex in nearby
                    )
                ]
                if not survivors:
                    continue
                cluster_items = survivors
            if not cluster_items:
                continue
            para = build_paragraph(
                cluster_items, W, H, None,
                cfg.pipeline.layout_overlap_ratio, cfg.pipeline.layout_gap_ratio,
            )
            if para is not None:
                paragraphs.append(para)
        return paragraphs

    # -- main entry points -----------------------------------------------------
    def process_pil(self, img: Image.Image) -> dict:
        img = img.convert("RGB")
        W, H = img.size
        cfg = self.cfg
        boxes, _scores = self.region.detect_pil(img)
        paragraphs: list[dict] = []

        if len(boxes) == 0 and cfg.lines.run_lines_on_full_image_if_no_regions and cfg.lines.enable_line_stage:
            log.info("No regions; running line detection on full image")
            quads = self.lines.detect_pil(self._cleanup_pil(img))
            items: list[dict] = []
            min_px = float(cfg.lines.min_line_px)
            use_ppocr = cfg.rec.backend == "ppocr"
            img_bgr = None
            if quads:
                img_bgr = np.array(img.convert("RGB"))[:, :, ::-1]
            for quad in quads:
                q = np.asarray(quad, dtype=np.float32)
                det_xyxy = polygon_to_xyxy(q)  # tight box for furigana rules
                sort_wh = quad_true_size(q)  # deskewed size for reading order
                if use_ppocr:
                    assert img_bgr is not None
                    text, xyxy = self._recognize_quad(img_bgr, q)
                    if not text:
                        continue
                    items.append(
                        {"text": text, "xyxy": xyxy, "det_xyxy": det_xyxy,
                         "quad": q.tolist(), "sort_wh": sort_wh})
                    continue
                x1, y1, x2, y2 = det_xyxy
                if min_px > 0 and max(x2 - x1, y2 - y1) < min_px:
                    continue
                assert img_bgr is not None
                text, xyxy = self._recognize_warped_quad(img_bgr, q)
                if not text:
                    continue
                items.append({"text": text, "xyxy": xyxy, "det_xyxy": det_xyxy,
                              "quad": q.tolist(), "sort_wh": sort_wh})
            for item in self._maybe_defurigana(items):
                para = build_paragraph(
                    [item], W, H, None,
                    cfg.pipeline.layout_overlap_ratio, cfg.pipeline.layout_gap_ratio,
                )
                if para is not None:
                    paragraphs.append(para)
            return self._wrap(W, H, self._finalize_paragraphs(paragraphs))

        workers = resolve_workers(cfg.general.workers)
        box_list = [tuple(float(v) for v in b) for b in boxes]
        expanded = self._expand_region_boxes(box_list, W, H)
        if self.allow_inner_parallel and workers > 1 and len(box_list) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(workers, len(box_list))) as pool:
                region_results = list(pool.map(lambda b: self._process_region(img, b), expanded))
        else:
            region_results = [self._process_region(img, b) for b in expanded]
        paragraphs: list[dict] = []
        existing_line_items: list[dict] = []
        for (rx1, ry1, rx2, ry2), line_items in region_results:
            if not line_items:
                continue
            line_items = self._maybe_defurigana(line_items)
            if not line_items:
                continue
            existing_line_items.extend(line_items)
            para = build_paragraph(
                line_items, W, H,
                (float(rx1), float(ry1), float(rx2), float(ry2)),
                cfg.pipeline.layout_overlap_ratio, cfg.pipeline.layout_gap_ratio,
            )
            if para is not None:
                paragraphs.append(para)
        paragraphs.extend(self._orphan_paragraphs(img, existing_line_items))
        return self._wrap(W, H, self._finalize_paragraphs(paragraphs))

    def process_image(self, image: Union[str, Path, bytes, bytearray, Image.Image]) -> dict:
        if isinstance(image, (bytes, bytearray)):
            return self.process_pil(load_pil(bytes(image)))
        return self.process_pil(load_pil(image))

    def process_bytes(self, data: bytes) -> dict:
        return self.process_pil(load_pil(data))

    @staticmethod
    def _wrap(W: int, H: int, paragraphs: list[dict]) -> dict:
        return {
            "image_properties": {"width": W, "height": H},
            "engine_capabilities": dict(ENGINE_CAPABILITIES),
            "paragraphs": paragraphs,
        }
