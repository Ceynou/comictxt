# Benchmarks

Staged tuning against `ground_truth/` (12 mokuro-style pages, 109 blocks /
217 lines). Each stage freezes the previous ones:

1. `bench_rec.py` — recognition only (GT line quads, no detection)
2. `bench_lines.py` — line detection (DB post-params + size gates),
   furigana-aware FP buckets
3. `bench_region.py` — region detection (YOLO size x conf x imgsz x
   containment), validated on both backends
4. full-pipeline `comictxt eval` A/B (old vs new defaults)

Results live in `results/*.json` (+ `.log`). Run everything with the repo
venv python. `viz_lines.py` draws detections vs GT for one page.

## 1. Recognition (text-only, GT quads)

Hayai OCR v2.5 Nova, CPU, greedy, 217 lines (184 text / 33 SFX by
font_size>=60), crops warped exactly like the pipeline:

| line_pad | max_num_patches | text CER | all CER | all EM | ms/line |
|---:|---:|---:|---:|---:|---:|
| 0 | 256 | 0.0792 | 0.0861 | 0.8433 | 567 |
| 0 | 384 | 0.0740 | 0.0837 | 0.8341 | 795 |
| **0** | **512** | **0.0691** | **0.0806** | 0.8341 | 1131 |
| 2 | 256 | 0.0723 | 0.0815 | 0.8203 | 628 |
| 2 | 512 | 0.0822 | 0.0898 | 0.8249 | 1099 |
| 4 | 512 | 0.0839 | 0.0919 | 0.8203 | 1105 |

**Winner: `rec.line_pad=0`, `rec.max_num_patches=512`.** Padding always
hurts (neighbor glyphs/artwork bleed into the crop); more patches =
monotonically better text CER (matches the model card).

## 2. Line detection (gt-crop context = per-block crops, full = whole page)

Metric: box IoU>=0.5 greedy match vs GT lines; FPs bucketed into
ruby-like (small, thin, beside a bigger aligned line — furigana-shaped)
vs other. Sweep: 3 scales x 4 thresh x 4 box_thresh x 4 unclip + refine.

Findings:

- DB post-params alone plateau at F1 ~0.76 (R 0.908); ~85/101 FPs are
  ruby-like and insensitive to thresh/box_thresh/unclip/scale.
- `unclip_ratio` 1.8 > 1.4 (higher matched IoU, same recall).
- The decisive knob is `min_short_side` (min side of the unclipped box):

| min_short_side | gt-crop F1 | R | rubyFP | otherFP | full-page F1 |
|---:|---:|---:|---:|---:|---:|
| 6 (old) | 0.761 | 0.908 | 87 | 17 | 0.743 |
| 12 | 0.822 | 0.912 | 52 | 15 | 0.775 |
| **18** | **0.868** | 0.876 | **19** | 12 | **0.860** |
| 24 | 0.793 | 0.714 | 7 | 12 | 0.791 |
| 30 | 0.587 | 0.442 | 3 | 11 | 0.610 |

**Winner: 960 / thresh 0.20 / box 0.25 / unclip 1.8 / min_short_side 18**
(F1 0.863 gt-crop + 0.860 full, rubyFP 87->19; 24+ starts killing small
text pages). thresh/box_thresh/scale are second-order at the knee.

## 3. Region detection (block F1 @ IoU 0.5, onnx sweep + ultralytics check)

| size@imgsz | conf | contain | onnx F1 | ultralytics F1 |
|---|---|---|---:|---:|
| **m@640** | **0.40** | **merge@0.8** | **0.889** | 0.884 |
| x@640 | 0.20 | merge@0.8 | 0.885 | 0.885 |
| s@640 | 0.30 | merge@0.8 | 0.883 | — |
| l@640 | 0.40 | merge@0.8 | 0.873 | — |
| n@960 | 0.20 | merge@0.8 | 0.872 | — |

- imgsz 960/1280 never beat 640; merge@0.8 > drop@0.8 > keep; NMS on/off
  irrelevant once containment-merge runs (confirms the earlier ablation).
- **Winner: model_size m, conf 0.40, imgsz 640, merge@0.8** — same F1 as x
  at ~2.7x less compute.

## 4. Full pipeline (`comictxt eval ground_truth/`, 12 pages / 109 blocks)

| config | P | R | vert | CER | matched |
|---|---|---|---|---|---|
| old defaults (l/0.20, 0.15/1.4/6, pad2/256) | 0.877 | 0.917 | 0.990 | 0.0905 | 100/109 |
| new det/lines + old rec (pad2/256) | 0.896 | 0.945 | 0.990 | 0.1019 | 103/109 |
| **new defaults (m/0.40, 0.20/1.8/18, pad0/512)** | **0.888** | **0.945** | 0.990 | **0.0924** | **103/109** |

Row 2 vs 3 isolates recognition on the same 103 matched blocks: pad0/512
beats pad2/256 through the pipeline too (CER 0.0924 vs 0.1019). The old
run's lower CER (0.0905) is a different, easier 100-block population.
