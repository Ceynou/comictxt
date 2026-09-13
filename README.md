# comictxt

Manga/comic OCR pipeline:

1. **Region detection** — AnimeText YOLO (`deepghs/AnimeText_yolo`, default size `x`), with nested-region suppression
2. **Line detection** — PP-OCRv6 manga det (`Kellenok/PP-OCRv6_manga`, optional per-region), with noise gates + furigana filter
3. **Text recognition** — Hayai OCR v2, ONNX (`JustANormalTinkerer/hayai-ocr-v2-onnx`, default) or PyTorch (`JustANormalTinkerer/hayai-ocr-v2`), or PP-OCRv6 manga CTC (`Kellenok/PP-OCRv6_manga`, `--set rec.backend=ppocr`)

Outputs **owocr-compatible JSON** (`image_properties` + `paragraphs`/`lines`/`bounding_box` normalized) so it works as a drop-in backend for [neokuro](https://github.com/kamperemu/neokuro) / mokuro generation.

## Install

```bash
pip install -e .
# optional ultralytics backend for region detection:
pip install -e ".[ultralytics]"
# optional PyTorch backend for recognition:
pip install -e ".[torch]"
```

System deps are pure pip (`onnxruntime`, `opencv-python-headless`, `tokenizers`, `pyclipper`, `websockets`, `pydantic`).

## Quick start

```bash
# single image -> JSON stdout
comictxt infer page.jpg

# single image -> file
comictxt infer page.jpg -o page.json --pretty

# batch directory (threaded by default, auto workers)
comictxt infer ./pages/ -o ./out/ --pretty

# batch with 4 processes (each loads its own models)
comictxt infer ./pages/ -o ./out/ --workers 4 --workers-mode processes

# custom config + overrides
comictxt infer page.jpg --config configs/default.toml --set region.conf=0.35 --no-lines

# reading order is on by default; disable with:
comictxt infer page.jpg --no-reorder  # or --no-reorder-blocks / --no-reorder-lines

# PP-OCR recognizer (Kellenok Space port, CTC) instead of Hayai
comictxt infer page.jpg --set rec.backend=ppocr

# optional image cleanup before detection/recognition (off by default)
comictxt infer page.jpg --set preprocess.enable=true --set preprocess.black_point=40 --set preprocess.white_point=210 --set preprocess.sharpen=0.5

# fully offline (local HF cache only; also via HF_HUB_OFFLINE=1)
comictxt infer page.jpg --offline

# PyTorch recognizer instead of ONNX
comictxt infer page.jpg --set rec.backend=torch

# serve owocr-compatible websocket (neokuro connects here)
comictxt serve --port 7331
neokuro /path/to/vol --port 7331

# evaluate against mokuro ground truth
comictxt eval ground_truth/ --pretty -o eval/results.json

# per-stage debug report (annotated PNGs + values + index.html):
# region conf/NMS/containment, padding, raw contour -> unclipped poly ->
# final quad, warp/trim + recognition crops, furigana decisions, paragraphs
comictxt debug page.jpg -o debug/page/

# configuration
comictxt config --print          # print shipped defaults
comictxt config --path           # show packaged / user / effective paths
comictxt config --init           # write defaults to ~/.config/comictxt/config.toml
```

## Pipeline details

## Pipeline details

Line detection, recognition warps/trims, furigana rules and reading order are
ported from Kellenok's [PP-OCR_manga Space](https://huggingface.co/spaces/Kellenok/PP-OCR_manga)
(`Kellenok/PP-OCRv6_manga`); the debug report (`comictxt debug`) shows every
stage with before/after images.

- **Line detection** (`lines.*`): white `det_margin` border (default 16px, subtracted after unclip), Space scale policy (`det_long_side` cap 960 / `det_min_side` floor 480, snap 32), box-score filter, pyclipper unclip of the largest path (`unclip_ratio`, default 1.4), 6px minimum (`min_short_side`), then `minAreaRect` + `box_pad` (default 4.0) 4-point quads.
- **Recognition** (`rec.backend = torch|onnx|ppocr`): `ppocr` runs the Space's CTC recognizer (`rec/manga_rec_v0.1.onnx` + `ppocrv6_dict.txt`) with perspective warp of the detector quad, Otsu-projection furigana/margin trim (`rec.ppocr_trim`) and height-48 CTC decode. Hayai backends (`onnx`/`torch`) keep the axis-aligned crop path with no rotation.
- **Preprocessing** (`[preprocess]`, off by default): levels stretch (`black_point`/`white_point`) + unsharp mask (`sharpen`), applied to line-detection inputs and recognition crops.
- **Orientation** is layout-based, not aspect-ratio based: neighboring lines with a small x-gap and strong y-overlap vote vertical (`TOP_TO_BOTTOM`); stacked lines vote horizontal. Single-line paragraphs fall back to the parent region-box aspect. Tunables: `pipeline.layout_overlap_ratio`, `pipeline.layout_gap_ratio`.
- **Reading order** (`pipeline.reorder_blocks` / `pipeline.reorder_lines`, both default on): Space column/row sort — vertical reads right-to-left columns (top-to-bottom within), horizontal reads top-to-bottom rows (left-to-right within) — applied to paragraphs (global area-weighted vote) and to lines within each paragraph. Disable with `--no-reorder`.
- **Empty boxes are dropped**: regions/lines with no OCR text never produce paragraphs.
- **Nested regions**: post-detection handling via `region.contain_thresh` / `region.contain_action` (`merge`|`drop`|`keep`, default `merge`; `keep` returns raw YOLO output) — IoU-NMS alone misses nested boxes, and score-ordered suppression fails because YOLO often scores the inner fragment higher. `region.nms` (default false, onnx backend only) re-enables IoU-NMS. Ablation on `ground_truth/` (onnx backend, size n, current detector): merge-only beats NMS+merge (F1 0.865 vs 0.838, CER 0.186 vs 0.200); NMS-only collapses precision (0.61 in the earlier ablation) and no suppression at all is computationally infeasible. Toggle: `--set region.nms=true`, `--set region.contain_action=keep`.
- **Noise gates**: lines smaller than `lines.min_line_px` are skipped before recognition, and nearly-flat crops (`rec.blank_std_thresh`) are skipped — both prevent recognizer hallucinations on specks/blank areas (ppocr/CTC emits empty strings on blanks, so it needs no blank gate).
- **Furigana filter** (`lines.furigana_*`, Space `is_furigana_pair` rules): ruby annotates kanji only, sits strictly right/above the main line, and obeys scale laws (32px cap, 70% thickness, length and char-size laws). Geometry is evaluated on the tight detection boxes (`det_xyxy`), not the `line_pad`-inflated recognition crops, which otherwise flip the thickness ratio for borderline ruby.

## Parallelism

- `general.workers` (default `0` = auto, `min(4, cpu)`), `general.workers_mode = threads|processes`.
- Threads: batch CLI fans out pages over one shared pipeline; single images fan out regions. ONNX sessions are thread-safe; recognizer caches are locked.
- Processes: batch CLI only — each worker loads its own models (`--workers-mode processes`).
- WS server: bounded worker pool (`server.max_workers`, default 2); extra connections queue.

## Eval

`comictxt eval <gt_dir>` runs the pipeline over `*.webp`+`*.json` mokuro pairs and reports block precision/recall (IoU≥0.5), writing-direction accuracy, and line CER. Current numbers on `ground_truth/` (009/010/083/131, region onnx size n): Hayai backend P=0.914 R=0.821 F1=0.865 CER=0.151; ppocr backend F1=0.865 CER=0.184.

## Python API

```python
from comictxt.config import ComictxtConfig
from comictxt.pipeline import ComicTxtPipeline

cfg = ComictxtConfig()  # everything has defaults; override via kwargs/TOML
pipe = ComicTxtPipeline(cfg)
result = pipe.process_image("page.jpg")  # owocr-compatible dict
```

## Config

Every parameter is configurable via `ComictxtConfig` dataclass, TOML file, `--set KEY=VALUE`, CLI flags, or Python kwargs. See `configs/default.toml` (also shipped inside the wheel as `comictxt/data/default.toml`).

Config file resolution: `--config` > `$COMICTXT_CONFIG` > `~/.config/comictxt/config.toml` > built-in defaults. On the first `infer`/`serve`/`debug`/`eval` run the user config is auto-created from shipped defaults and then read (edit it freely; refresh with `comictxt config --init --force`; if it can't be written, a warning is logged and built-in defaults are used). `config --print`/`--path` never create files, and an explicit `--config` pointing at a missing file is an error. If local model files are cached, `--offline` (or `HF_HUB_OFFLINE=1`) guarantees no network access.

## Websocket protocol (owocr-compatible)

Server listens on `host:port` (default `0.0.0.0:7331`). For each binary image message:

1. client: `send_binary(image_bytes)`
2. server: `send("True")` (ack)
3. server: `send(json_string)` (owocr result)

This matches `neokuro/owosocket.py::process_image` exactly.
