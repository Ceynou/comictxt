# comictxt

Manga/comic OCR pipeline:

1. **Region detection** — AnimeText YOLO (`deepghs/AnimeText_yolo`, default size `m` @ conf 0.40), with nested-region suppression
2. **Line detection** — PP-OCRv6 manga det (`Kellenok/PP-OCRv6_manga`): per-region pass plus a full-page **orphan sweep** that adopts lines the region flow missed (small SFX, captions, region-detector misses)
3. **Text recognition** — Hayai OCR v2.5 Nova (`JustANormalTinkerer/hayai-ocr-v2.5-nova`, PyTorch): SigLIP2 NaFlex + 12-layer decoder, handles vertical and stylized text natively

Outputs **owocr-compatible JSON** (`image_properties` + `paragraphs`/`lines`/`bounding_box` normalized) so it works as a drop-in backend for [neokuro](https://github.com/kamperemu/neokuro) / mokuro generation.

Recognition requires the `torch` extra (`transformers>=5`, `safetensors`); it is the only recognizer, so in practice install `pip install -e ".[torch]"` (or `all`).

## Install

```bash
pip install -e ".[torch]"
# optional ultralytics backend for region detection (default backend):
pip install -e ".[ultralytics]"
```

## Quick start

```bash
# single image -> JSON stdout
comictxt infer page.jpg

# single image -> file
comictxt infer page.jpg -o page.json --pretty

# batch directory (threaded by default, auto workers)
# inputs: jpg jpeg png webp avif jxl bmp gif tif tiff
comictxt infer ./pages/ -o ./out/ --pretty

# batch with 4 processes (each loads its own models)
comictxt infer ./pages/ -o ./out/ --workers 4 --workers-mode processes

# custom config + overrides
comictxt infer page.jpg --config configs/default.toml --set region.conf=0.35 --no-lines

# reading order is on by default; disable with:
comictxt infer page.jpg --no-reorder  # or --no-reorder-blocks / --no-reorder-lines

# optional image cleanup before detection/recognition (off by default)
comictxt infer page.jpg --set preprocess.enable=true --set preprocess.black_point=40 --set preprocess.white_point=210 --set preprocess.sharpen=0.5

# fully offline (local HF cache only; also via HF_HUB_OFFLINE=1)
comictxt infer page.jpg --offline

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
comictxt config                  # labeled paths + active source + effective TOML
comictxt config --print          # print shipped defaults
comictxt config --effective      # print effective TOML (active file + --config/--set)
comictxt config --path           # show labeled default / user / effective paths
comictxt config --init           # write defaults to ~/.config/comictxt/config.toml
```

## Pipeline details

Line detection, recognition warps/trims, furigana rules and reading order are
ported from Kellenok's [PP-OCR_manga Space](https://huggingface.co/spaces/Kellenok/PP-OCR_manga)
(`Kellenok/PP-OCRv6_manga`); the debug report (`comictxt debug`) shows every
stage with before/after images.

- **Line detection** (`lines.*`): white `det_margin` border (default 16px, subtracted after unclip), Space scale policy (`det_long_side` cap 960 / `det_min_side` floor 480, snap 32), box-score filter, pyclipper unclip of the largest path (`unclip_ratio`, default 1.8), minimum side (`min_short_side`, default 18px — the ruby-exclusion knee from the line ablation), then `minAreaRect` + `box_pad` (default 4.0) 4-point quads.
- **Orphan sweep** (`lines.orphan_sweep`, default on): after the per-region pass, the line detector runs once over the whole page; quads that duplicate already-recognized lines (IoU/center-in-box) are skipped, survivors are recognized (needing `lines.orphan_min_chars` alphanumeric characters — one-glyph junk dominates on raw artwork), clustered into paragraphs, and furigana-checked against nearby recognized lines (both orientations, so a junk neighbor cannot flip the vote). Recovers small SFX/captions the region stage missed entirely.
- **Recognition** (`rec.*`, Hayai OCR v2.5 Nova): perspective-warp each rotated detector quad to a tight upright crop (vertical stays vertical — no rotate-if-tall) with **no extra padding** (`rec.line_pad=0`: the text-only ablation shows any border context hurts CER), `min_crop_size` upscales degenerate crops, `max_num_patches=512` is the quality patch budget (256/384 for speed). `rec.torch_revision` selects a repo branch when the HF cache holds several; the resolver pins the exact commit via the cache refs.
- **Preprocessing** (`[preprocess]`, off by default): levels stretch (`black_point`/`white_point`) + unsharp mask (`sharpen`), applied to line-detection inputs and recognition crops.
- **Orientation** is layout-based, not aspect-ratio based: neighboring lines with a small x-gap and strong y-overlap vote vertical (`TOP_TO_BOTTOM`); stacked lines vote horizontal. Single-line paragraphs fall back to the parent region-box aspect. Tunables: `pipeline.layout_overlap_ratio`, `pipeline.layout_gap_ratio`.
- **Reading order** (`pipeline.reorder_blocks` / `pipeline.reorder_lines`, both default on): Space column/row sort — vertical reads right-to-left columns (top-to-bottom within), horizontal reads top-to-bottom rows (left-to-right within) — applied to paragraphs (global area-weighted vote) and to lines within each paragraph. Disable with `--no-reorder`.
- **Empty boxes are dropped**: regions/lines with no OCR text never produce paragraphs.
- **Region crop padding** (`region.pad_ratio`, `region.pad_mode`, default `auto`): elongated isolated regions expand uniformly — the smallest side grows by the same absolute amount as the biggest side — so tight long-line crops gain real context left/right instead of only along their length; when the widened box would reach another detected region (adjacent bubbles/columns) it falls back to per-axis proportional growth. `uniform`/`proportional`/`max` force one behavior.
- **Nested regions**: post-detection handling via `region.contain_thresh` / `region.contain_action` (`merge`|`drop`|`keep`, default `merge`; `keep` returns raw YOLO output) — IoU-NMS alone misses nested boxes, and score-ordered suppression fails because YOLO often scores the inner fragment higher. `region.nms` (default false, onnx backend only) re-enables IoU-NMS. Ablation on `ground_truth/` (onnx backend, size n, current detector): merge-only beats NMS+merge (F1 0.865 vs 0.838, CER 0.186 vs 0.200); NMS-only collapses precision (0.61 in the earlier ablation) and no suppression at all is computationally infeasible. Toggle: `--set region.nms=true`, `--set region.contain_action=keep`.
- **Noise gates**: lines smaller than `lines.min_line_px` are skipped before recognition, lines whose text is a single alphanumeric letter are dropped (`lines.min_line_chars` — junk fragments like `T`/`キ` on artwork; punctuation-only lines like `ー` always pass), and nearly-flat crops (`rec.blank_std_thresh`) are skipped — all prevent recognizer hallucinations on specks/blank areas.
- **Furigana filter** (`lines.furigana_*`, Space `is_furigana_pair` rules): ruby annotates kanji only, sits strictly right/above the main line, and obeys scale laws (length and char-size laws, thickness ratio `furigana_size_ratio` default 0.75). Geometry is evaluated on ink extents (`ink_wh`: warp + Otsu bbox of the detector quad) when available — det boxes carry `box_pad` inflation whose ratios wander 0.5–0.85 for true ruby; ink ratios are stable at ~0.5–0.6. The char-size law is evaluated first and, when it confirms ruby (≥2 chars at ~half the main's per-char size), the thickness law only requires the ruby to not be fatter (hand-drawn styles draw chunky ruby). Katakana ruby containing the long-vowel mark (エキスパート) is allowed; pure `ー` marks stay kept.

## Parallelism

- `general.workers` (default `0` = auto, `min(4, cpu)`), `general.workers_mode = threads|processes`.
- Threads: batch CLI fans out pages over one shared pipeline; single images fan out regions. ONNX sessions are thread-safe; recognizer caches are locked.
- Processes: batch CLI only — each worker loads its own models (`--workers-mode processes`).
- WS server: bounded worker pool (`server.max_workers`, default 2); extra connections queue.

## Eval

`comictxt eval <gt_dir>` runs the pipeline over `*.webp`+`*.json` mokuro pairs and reports block precision/recall (IoU≥0.5), writing-direction accuracy, and line CER. Numbers on all 12 `ground_truth/` pages (deliberately hard: tilted SFX, tiny ruby, adjacent columns), single-threaded, defaults tuned per stage in `benchmarks/`:

| config | P | R | vert | CER | matched |
|---|---|---|---|---|---|
| pre-tuning defaults (l/0.20, 0.15/1.4/6, pad2/256) | 0.877 | 0.917 | 0.990 | 0.0905 | 100/109 |
| tuned det/lines + old rec | 0.896 | 0.945 | 0.990 | 0.1019 | 103/109 |
| **defaults (m@640/0.40, lines 0.20/1.8/18, rec pad0/512)** | **0.888** | **0.945** | 0.990 | **0.0924** | **103/109** |

Stage-wise ablations (text-only recognition CER, furigana-aware line F1, region size/conf sweeps, backend parity) live in `benchmarks/README.md` with raw results under `benchmarks/results/`. Highlights: `rec.line_pad=0` + `max_num_patches=512` (text CER 0.069 on GT quads), `lines.min_short_side=18` is the ruby-exclusion knee (line F1 0.761→0.868, ruby FPs 87→19), region `m@640 conf 0.40 merge@0.8` (F1 0.889 onnx / 0.884 ultralytics — same as `x` at ~2.7x less compute). Known remaining ceilings: adjacent-column blocks merged at the region level (the two bubbles physically touch, no geometric gap to split on), split/fragmented stylized SFX (チーーーン, ザズ!!), and recognizer long-tail errors on heavy SFX.

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
