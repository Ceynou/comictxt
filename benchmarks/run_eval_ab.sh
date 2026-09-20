#!/bin/bash
cd /home/user/Projects/comictxt
PY=/home/user/.ocr/bin/python
$PY -m comictxt.cli eval ground_truth/ --config configs/default.toml --pretty \
  -o benchmarks/results/eval_new.json > benchmarks/results/eval_new.log 2>&1
$PY -m comictxt.cli eval ground_truth/ --config configs/default.toml --pretty \
  --set region.model_size=l --set region.conf=0.20 \
  --set lines.thresh=0.15 --set lines.unclip_ratio=1.4 --set lines.min_short_side=6 \
  --set rec.line_pad=2 --set rec.max_num_patches=256 \
  -o benchmarks/results/eval_old.json > benchmarks/results/eval_old.log 2>&1
echo DONE > benchmarks/results/eval_ab.done
