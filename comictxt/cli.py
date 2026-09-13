"""CLI: `comictxt infer ...`, `comictxt serve ...`, `comictxt debug ...`, `comictxt config ...`."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from comictxt.config import (
    ComictxtConfig,
    default_config_path,
    default_config_toml_text,
    ensure_user_config,
    parse_override_value,
    resolve_config_file,
    user_config_path,
)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".avif", ".jxl", ".bmp", ".gif", ".tif", ".tiff")

log = logging.getLogger("comictxt")


def _build_config(args) -> ComictxtConfig:
    return _load_effective_config(args, create_user_config=True)


def _load_effective_config(args, create_user_config: bool = False) -> ComictxtConfig:
    # Resolution order: explicit --config > $COMICTXT_CONFIG > user config >
    # built-in defaults. An explicit --config that does not exist is an error.
    # On first start (no file anywhere), the user config is auto-created from
    # shipped defaults and then read, when create_user_config is set (all
    # commands except `config --print/--path`, which never create files).
    explicit = getattr(args, "config", None)
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise SystemExit(f"Config file not found: {p}")
        cfg = ComictxtConfig.from_toml(p)
    else:
        found = resolve_config_file(None)
        if found is None and create_user_config:
            created = ensure_user_config()
            if created is not None:
                log.info("Created default user config at %s", created)
                found = resolve_config_file(None)
                if found is None and created.is_file():
                    found = created
            else:
                log.warning(
                    "Could not write user config at %s; using built-in defaults",
                    user_config_path(),
                )
        cfg = ComictxtConfig.from_toml(found) if found is not None else ComictxtConfig()
    overrides: dict = {}
    for item in getattr(args, "set", None) or []:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got: {item!r}")
        k, v = item.split("=", 1)
        overrides[k.strip()] = parse_override_value(v.strip())
    # convenience shortcuts
    shortcuts = {
        "host": "server.host",
        "port": "server.port",
        "max_workers": "server.max_workers",
        "region_backend": "region.backend",
        "region_model_size": "region.model_size",
        "region_conf": "region.conf",
        "region_imgsz": "region.imgsz",
        "precision": "rec.precision",
        "rec_backend": "rec.backend",
        "workers": "general.workers",
        "workers_mode": "general.workers_mode",
        "no_lines": "lines.enable_line_stage",
        "offline": "general.offline",
    }
    for attr, key in shortcuts.items():
        val = getattr(args, attr, None)
        if val is None:
            continue
        if attr == "no_lines" and val:
            overrides[key] = False
        elif attr == "no_lines":
            continue
        else:
            overrides[key] = val
    # --reorder-blocks / --no-reorder-blocks (BooleanOptionalAction, default None)
    for attr, key in (
        ("reorder_blocks", "pipeline.reorder_blocks"),
        ("reorder_lines", "pipeline.reorder_lines"),
    ):
        val = getattr(args, attr, None)
        if val is not None:
            overrides[key] = val
    if getattr(args, "no_reorder", False):
        overrides["pipeline.reorder_blocks"] = False
        overrides["pipeline.reorder_lines"] = False
    if overrides:
        cfg = cfg.with_overrides(overrides)
    return cfg


def _iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _cmd_infer(args) -> int:
    cfg = _build_config(args)
    logging.basicConfig(
        level=getattr(logging, cfg.general.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    src = Path(args.input)
    if not src.exists():
        print(f"Input not found: {src}", file=sys.stderr)
        return 2
    images = _iter_images(src)
    if not images:
        print(f"No images found in: {src}", file=sys.stderr)
        return 2

    out = Path(args.output) if args.output else None
    if out is not None and src.is_dir():
        out.mkdir(parents=True, exist_ok=True)

    from comictxt.config import resolve_workers

    workers = getattr(args, "workers", None)
    if workers is None:
        workers = resolve_workers(cfg.general.workers)
    mode = getattr(args, "workers_mode", None) or cfg.general.workers_mode
    if len(images) < 2:
        workers = 1

    try:
        if workers > 1 and mode == "processes":
            results = _infer_batch_processes(cfg, images)
        elif workers > 1:
            results = _infer_batch_threads(cfg, images, workers)
        else:
            from comictxt.pipeline import ComicTxtPipeline

            pipe = ComicTxtPipeline(cfg)
            results = [(p, _safe_process(pipe, p)) for p in images]
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130

    rc = 0
    for img_path, result in results:
        if isinstance(result, Exception):
            log.error("Failed %s: %s", img_path.name, result)
            rc = 1
            continue
        text = json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None)
        if out is None:
            if len(images) > 1:
                print(f"===== {img_path.name} =====")
            print(text)
        elif out.suffix.lower() == ".json" and src.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            log.info("Wrote %s", out)
        else:
            dest = out if out.suffix.lower() == ".json" else (out / (img_path.stem + ".json"))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
            log.info("Wrote %s", dest)
    return rc


def _safe_process(pipe, img_path):
    try:
        log.info("Processing %s", img_path.name)
        return pipe.process_image(img_path)
    except Exception as e:  # noqa: BLE001 - per-page isolation, batch continues
        return e


def _infer_batch_threads(cfg, images: list[Path], workers: int):
    from concurrent.futures import ThreadPoolExecutor

    from comictxt.pipeline import ComicTxtPipeline

    # inner fan-out disabled: the outer pool already parallelizes
    pipe = ComicTxtPipeline(cfg, allow_inner_parallel=False)
    pipe.warmup()
    pool = ThreadPoolExecutor(max_workers=min(workers, len(images)))
    try:
        return list(zip(images, pool.map(lambda p: _safe_process(pipe, p), images)))
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


_PROC_PIPE = None


def _proc_init(cfg_dict: dict):
    global _PROC_PIPE
    from comictxt.config import ComictxtConfig
    from comictxt.pipeline import ComicTxtPipeline

    _PROC_PIPE = ComicTxtPipeline(ComictxtConfig(**cfg_dict), allow_inner_parallel=False)
    _PROC_PIPE.warmup()


def _proc_run(path_str: str):
    from pathlib import Path as _P

    p = _P(path_str)
    try:
        return (path_str, _PROC_PIPE.process_image(p))
    except Exception as e:  # noqa: BLE001
        return (path_str, e)


def _infer_batch_processes(cfg, images: list[Path]):
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path as _P

    from comictxt.config import resolve_workers

    workers = resolve_workers(cfg.general.workers)
    pool = ProcessPoolExecutor(
        max_workers=min(workers, len(images)),
        initializer=_proc_init,
        initargs=(cfg.model_dump(),),
    )
    try:
        raw = list(pool.map(_proc_run, [str(p) for p in images]))
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return [(_P(s), r) for s, r in raw]


def _cmd_serve(args) -> int:
    cfg = _build_config(args)
    from comictxt.server import serve_forever

    serve_forever(cfg)
    return 0


def _cmd_eval(args) -> int:
    import logging

    cfg = _build_config(args)
    logging.basicConfig(
        level=getattr(logging, cfg.general.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    from comictxt.eval import run_eval
    from comictxt.pipeline import ComicTxtPipeline

    gt_dir = Path(args.gt_dir)
    try:
        summary = run_eval(gt_dir, ComicTxtPipeline(cfg), iou_thresh=args.iou)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    text = json.dumps(summary, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        log.info("Wrote %s", args.output)
    else:
        print(text)
    return 0


def _cmd_debug(args) -> int:
    cfg = _build_config(args)
    logging.basicConfig(
        level=getattr(logging, cfg.general.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    from comictxt.debug import run_debug

    src = Path(args.input)
    if not src.exists():
        print(f"Input not found: {src}", file=sys.stderr)
        return 2
    images = _iter_images(src)
    if not images:
        print(f"No images found in: {src}", file=sys.stderr)
        return 2
    out_dir = Path(args.output)
    try:
        for img_path in images:
            dest = out_dir / img_path.stem if len(images) > 1 else out_dir
            index = run_debug(img_path, dest, cfg)
            log.info("Wrote %s", index)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    return 0


def _cmd_config(args) -> int:
    if args.print:
        print(default_config_toml_text(), end="" if default_config_toml_text().endswith("\n") else "\n")
        return 0
    if args.path:
        print(default_config_path())
        print(user_config_path())
        found = resolve_config_file(getattr(args, "config", None))
        print(found if found is not None else "(no user config found; using built-in defaults)")
        return 0
    if args.init:
        dest = user_config_path()
        if dest.is_file() and not args.force:
            print(f"User config already exists: {dest} (use --force to overwrite)", file=sys.stderr)
            return 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(default_config_toml_text(), encoding="utf-8")
        print(f"Wrote {dest}")
        return 0
    # default: show paths + effective source
    print(default_config_path())
    print(user_config_path())
    return 0


def _add_common_options(parser: argparse.ArgumentParser, is_sub: bool = False) -> None:
    # Subparsers use SUPPRESS so a top-level --config/--set is not clobbered
    # by subcommand defaults when the flag is only given once.
    default_config = argparse.SUPPRESS if is_sub else None
    default_set = argparse.SUPPRESS if is_sub else None
    parser.add_argument("--config", type=Path, default=default_config, help="TOML config file (default: built-in defaults)")
    parser.add_argument(
        "--set",
        action="append",
        default=default_set,
        metavar="KEY=VALUE",
        help="Override any config value, e.g. --set region.conf=0.35 --set lines.enable_line_stage=false. Repeatable.",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="comictxt", description="Manga OCR: region->lines->recognition")
    _add_common_options(p)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("infer", help="Run OCR on an image or directory")
    _add_common_options(pi, is_sub=True)
    pi.add_argument("input", type=Path, help="Image file or directory of images")
    pi.add_argument("-o", "--output", type=Path, default=None, help="JSON file (single input) or output dir; omit for stdout")
    pi.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    pi.add_argument("--region-backend", choices=["onnx", "ultralytics"], default=None)
    pi.add_argument("--region-model-size", choices=["n", "s", "m", "l", "x"], default=None)
    pi.add_argument("--region-conf", type=float, default=None)
    pi.add_argument("--region-imgsz", type=int, default=None)
    pi.add_argument("--precision", choices=["fp32", "fp16", "quant"], default=None)
    pi.add_argument("--rec-backend", choices=["onnx", "torch", "ppocr"], default=None)
    pi.add_argument("--workers", type=int, default=None, help="Batch/region parallel workers (default: auto)")
    pi.add_argument("--workers-mode", choices=["threads", "processes"], default=None)
    pi.add_argument("--no-lines", action="store_true", help="Skip line detection; recognize each region directly")
    pi.add_argument("--offline", action="store_true", default=None, help="Offline mode: use only locally cached models (also via HF_HUB_OFFLINE=1)")
    pi.add_argument("--reorder-blocks", action=argparse.BooleanOptionalAction, default=None, help="Reorder blocks in reading order (default: on)")
    pi.add_argument("--reorder-lines", action=argparse.BooleanOptionalAction, default=None, help="Reorder lines in reading order (default: on)")
    pi.add_argument("--no-reorder", action="store_true", help="Disable both block and line reordering")
    pi.set_defaults(func=_cmd_infer)

    ps = sub.add_parser("serve", help="Host owocr-compatible websocket server")
    _add_common_options(ps, is_sub=True)
    ps.add_argument("--host", type=str, default=None)
    ps.add_argument("--port", type=int, default=None)
    ps.add_argument("--max-workers", type=int, default=None)
    ps.set_defaults(func=_cmd_serve)

    pe = sub.add_parser("eval", help="Evaluate against mokuro ground-truth JSONs")
    _add_common_options(pe, is_sub=True)
    pe.add_argument("gt_dir", type=Path, help="Directory with *.json GT + paired images")
    pe.add_argument("-o", "--output", type=Path, default=None)
    pe.add_argument("--pretty", action="store_true")
    pe.add_argument("--iou", type=float, default=0.5)
    pe.set_defaults(func=_cmd_eval)

    pd = sub.add_parser("debug", help="Dump per-stage annotated images + values + HTML report")
    _add_common_options(pd, is_sub=True)
    pd.add_argument("input", type=Path, help="Image file or directory of images")
    pd.add_argument("-o", "--output", type=Path, required=True, help="Output directory for the debug report")
    pd.add_argument("--offline", action="store_true", default=None, help="Offline mode: use only locally cached models")
    pd.add_argument("--reorder-blocks", action=argparse.BooleanOptionalAction, default=None)
    pd.add_argument("--reorder-lines", action=argparse.BooleanOptionalAction, default=None)
    pd.add_argument("--no-reorder", action="store_true", help="Disable both block and line reordering")
    pd.set_defaults(func=_cmd_debug)

    pc = sub.add_parser("config", help="Show or initialize configuration")
    _add_common_options(pc, is_sub=True)
    pc.add_argument("--print", action="store_true", help="Print the shipped default TOML")
    pc.add_argument("--path", action="store_true", help="Show packaged default, user config, and effective config paths")
    pc.add_argument("--init", action="store_true", help="Write shipped defaults to the user config path")
    pc.add_argument("--force", action="store_true", help="Overwrite existing user config with --init")
    pc.set_defaults(func=_cmd_config)
    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        raise SystemExit(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
