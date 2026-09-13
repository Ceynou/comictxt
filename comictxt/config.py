"""Central configuration. Every pipeline parameter is configurable via
Python kwargs, TOML file, or CLI --section.key overrides.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

YOLO_SIZES = ("n", "s", "m", "l", "x")

# Per-size F1-optimal thresholds from AnimeText model cards (threshold.json).
# Used when region.conf <= 0 (auto mode).
DEFAULT_CONF_BY_SIZE = {
    "n": 0.251,
    "s": 0.272,
    "m": 0.299,
    "l": 0.426,
    "x": 0.425,
}


def _hf_hub_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env:
        roots.append(Path(env))
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    # Also check HF_HOME layout
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    return roots


def _find_snapshot_file(repo_id_dashed: str, relative: str) -> Optional[Path]:
    """Search HF hub cache for snapshots/<sha>/<relative>."""
    for root in _hf_hub_roots():
        base = root / repo_id_dashed
        snaps = base / "snapshots"
        if not snaps.is_dir():
            continue
        for sha_dir in sorted(snaps.iterdir()):
            cand = sha_dir / relative
            if cand.exists():
                return cand
    return None


def _find_snapshot_dir(repo_id_dashed: str, relative_dir: str) -> Optional[Path]:
    for root in _hf_hub_roots():
        base = root / repo_id_dashed
        snaps = base / "snapshots"
        if not snaps.is_dir():
            continue
        for sha_dir in sorted(snaps.iterdir()):
            cand = sha_dir / relative_dir
            if cand.is_dir():
                return cand
    return None


def resolve_region_onnx(model_size: str, explicit: str = "") -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"region onnx_path not found: {p}")
        return p
    rel = f"yolo12{model_size}_animetext/model.onnx"
    found = _find_snapshot_file("models--deepghs--AnimeText_yolo", rel)
    if found is None:
        raise FileNotFoundError(
            f"Could not auto-resolve AnimeText YOLO size '{model_size}' ({rel}) "
            f"in HF cache. Set region.onnx_path explicitly."
        )
    return found


def resolve_region_pt(model_size: str, explicit: str = "") -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"region pt_path not found: {p}")
        return p
    rel = f"yolo12{model_size}_animetext/model.pt"
    found = _find_snapshot_file("models--deepghs--AnimeText_yolo", rel)
    if found is None:
        raise FileNotFoundError(
            f"Could not auto-resolve AnimeText YOLO .pt size '{model_size}' ({rel}) "
            f"in HF cache. Set region.pt_path explicitly."
        )
    return found


def resolve_region_conf(model_size: str, conf: float) -> float:
    """conf <= 0 means auto (per-size threshold.json, else fallback table)."""
    if conf > 0:
        return conf
    rel = f"yolo12{model_size}_animetext/threshold.json"
    found = _find_snapshot_file("models--deepghs--AnimeText_yolo", rel)
    if found is not None:
        try:
            import json

            data = json.loads(found.read_text())
            return float(data.get("threshold", DEFAULT_CONF_BY_SIZE.get(model_size, 0.4)))
        except Exception:
            pass
    return DEFAULT_CONF_BY_SIZE.get(model_size, 0.4)


def resolve_lines_model(explicit: str = "", use_fp16: bool = False) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"lines model_path not found: {p}")
        return p
    name = "manga_det_v0.1_fp16.onnx" if use_fp16 else "manga_det_v0.1.onnx"
    found = _find_snapshot_file("models--Kellenok--PP-OCRv6_manga", f"det/{name}")
    if found is None and use_fp16:
        # fall back to fp32
        found = _find_snapshot_file("models--Kellenok--PP-OCRv6_manga", "det/manga_det_v0.1.onnx")
    if found is None:
        raise FileNotFoundError(
            f"Could not auto-resolve PP-OCR det model ({name}) in HF cache. "
            f"Set lines.model_path explicitly."
        )
    return found


def resolve_rec_onnx_dir(explicit: str = "") -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_dir():
            raise FileNotFoundError(f"rec onnx_dir not found: {p}")
        return p
    found = _find_snapshot_dir(
        "models--JustANormalTinkerer--hayai-ocr-v2-onnx", "onnx"
    )
    if found is None:
        raise FileNotFoundError(
            "Could not auto-resolve Hayai onnx dir in HF cache. Set rec.onnx_dir explicitly."
        )
    return found


class RegionConfig(BaseModel):
    model_size: Literal["n", "s", "m", "l", "x"] = "x"
    backend: Literal["onnx", "ultralytics"] = "ultralytics"
    onnx_path: str = ""
    pt_path: str = ""
    imgsz: Union[int, list[int]] = 640
    conf: float = 0.16
    iou: float = 0.7
    max_det: int = 300
    # IoU-NMS on the onnx backend (ultralytics always runs its internal NMS).
    # Ablation (ground_truth/, new detector): merge-only beats NMS+merge
    # (F1 0.865 vs 0.838, CER 0.186 vs 0.200), so NMS defaults OFF and
    # containment-merge does the dedup. Set true to re-enable NMS.
    nms: bool = False
    # nested-box handling (catches nested regions that IoU-NMS misses):
    # "merge" unions a contained fragment into its container (score-order
    # independent); "drop" removes fragments inside a higher-scored box;
    # "keep" disables. contain_thresh<=0 disables.
    contain_thresh: float = 0.80
    contain_action: Literal["drop", "keep", "merge"] = "merge"
    pad_ratio: float = 0.15
    providers: list[str] = Field(default_factory=lambda: ["CPUExecutionProvider"])
    device: str = "cpu"

    def resolved_imgsz(self) -> tuple[int, int]:
        if isinstance(self.imgsz, int):
            return (self.imgsz, self.imgsz)
        if len(self.imgsz) == 1:
            return (int(self.imgsz[0]), int(self.imgsz[0]))
        return (int(self.imgsz[0]), int(self.imgsz[1]))

    def resolved_conf(self) -> float:
        return resolve_region_conf(self.model_size, self.conf)

    def resolved_onnx(self) -> Path:
        return resolve_region_onnx(self.model_size, self.onnx_path)

    def resolved_pt(self) -> Path:
        return resolve_region_pt(self.model_size, self.pt_path)


class LinesConfig(BaseModel):
    enable_line_stage: bool = True
    fallback_to_region_text: bool = True
    run_lines_on_full_image_if_no_regions: bool = False
    model_path: str = ""
    use_fp16: bool = False
    # Space scale policy: long side capped at det_long_side, inputs smaller
    # than det_min_side upscaled to it, snapped to multiples of 32 (min 64).
    det_long_side: int = 960
    det_min_side: int = 480
    # White-border margin (px) around the detection input: text touching the
    # crop edge still detects (subtracted after unclip, like the Space app).
    det_margin: int = 16
    thresh: float = 0.15
    box_thresh: float = 0.25
    unclip_ratio: float = 1.4
    # Minimum box side (image px) of the unclipped polygon; smaller kills
    # speck/noise detections before the (hallucination-prone) recognizer.
    min_short_side: int = 6
    # Safety padding (px, net scale) added around the minAreaRect before the
    # final 4-point quad (Space: +4.0).
    box_pad: float = 4.0
    max_candidates: int = 3000
    providers: list[str] = Field(default_factory=lambda: ["CPUExecutionProvider"])
    # skip detected lines whose longest side is smaller than this (image px).
    min_line_px: float = 12.0
    # furigana (ruby) filter, ported from Kellenok's PP-OCR_manga Space app
    furigana_filter: bool = True
    furigana_size_ratio: float = 0.70
    furigana_proximity_ratio: float = 0.35
    furigana_overlap_ratio: float = 0.05
    furigana_max_thickness_px: float = 32.0
    furigana_length_ratio: float = 1.05
    furigana_char_size_ratio: float = 0.85
    furigana_proximity_min: float = 8.0
    furigana_proximity_max: float = 16.0
    furigana_max_chars: int = 8

    def resolved_model(self) -> Path:
        return resolve_lines_model(self.model_path, self.use_fp16)


class RecConfig(BaseModel):
    backend: Literal["onnx", "torch", "ppocr"] = "torch"
    onnx_dir: str = ""
    precision: Literal["fp32", "fp16", "quant"] = "fp32"
    providers: list[str] = Field(default_factory=lambda: ["CPUExecutionProvider"])
    max_new_tokens: int = 128
    line_pad: int = 3
    min_crop_size: int = 8
    # skip crops that are nearly flat (grayscale std below this): blank bubble
    # areas make the recognizer hallucinate. <=0 disables.
    blank_std_thresh: float = 5.0
    # torch backend (@hayaiocr_rec): explicit snapshot dir ("" = auto-resolve)
    torch_model: str = ""
    torch_processor: str = ""
    device: str = "cpu"
    dtype: Literal["float32", "float16"] = "float32"
    # NaFlex patches per crop (256 standard; 384/512 dense panels per card)
    max_num_patches: int = 256
    # ppocr backend (Kellenok PP-OCRv6 manga rec): explicit model/dict paths
    # ("" = auto-resolve from HF cache). trim enables the Otsu-projection
    # furigana/margin trim on warped crops (Space: get_rotate_crop_image).
    ppocr_model: str = ""
    ppocr_dict: str = ""
    ppocr_use_fp16: bool = False
    ppocr_trim: bool = True

    def resolved_onnx_dir(self) -> Path:
        return resolve_rec_onnx_dir(self.onnx_dir)

    def resolved_torch_model(self) -> Path:
        from comictxt.rec_hayai_torch import resolve_rec_torch

        return resolve_rec_torch(self.torch_model)

    def resolved_ppocr_model(self) -> Path:
        from comictxt.rec_ppocr import resolve_ppocr_model

        return resolve_ppocr_model(self.ppocr_model, self.ppocr_use_fp16)

    def resolved_ppocr_dict(self) -> Path:
        from comictxt.rec_ppocr import resolve_ppocr_dict

        return resolve_ppocr_dict(self.ppocr_dict)


class PipelineConfig(BaseModel):
    layout_overlap_ratio: float = 0.5
    layout_gap_ratio: float = 0.75
    clip_to_image: bool = True
    # Reading-order reorder at the end of the pipeline (Space column/row
    # sort). reorder_blocks orders paragraphs (columns right->left when
    # vertical, rows top->bottom when horizontal); reorder_lines sorts lines
    # inside each paragraph the same way. Both default ON.
    reorder_blocks: bool = True
    reorder_lines: bool = True


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 7331
    max_size: int = 1_000_000_000
    # concurrent OCR requests; extra connections queue instead of oversubscribing
    max_workers: int = 2


class GeneralConfig(BaseModel):
    log_level: str = "INFO"
    # parallel workers for batch CLI / region fan-out. <=0 = auto (min(4, cpu)).
    workers: int = 0
    workers_mode: Literal["threads", "processes"] = "threads"
    # offline mode: never hit the network; use only locally cached model files.
    # Also honored via HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 env vars.
    offline: bool = False


def resolve_workers(workers: int) -> int:
    if workers > 0:
        return workers
    import os

    return max(1, min(4, os.cpu_count() or 4))


class PreprocessConfig(BaseModel):
    """Optional image cleanup before detection/recognition (off by default).

    Levels stretch: values <= black_point become 0, >= white_point become
    255 (increases black, decreases white — "leveling"). sharpen is an
    unsharp-mask amount (0 disables). Applied to line-detection inputs and
    recognition crops.
    """

    enable: bool = False
    black_point: int = 0
    white_point: int = 255
    sharpen: float = 0.0


class ComictxtConfig(BaseModel):
    region: RegionConfig = Field(default_factory=RegionConfig)
    lines: LinesConfig = Field(default_factory=LinesConfig)
    rec: RecConfig = Field(default_factory=RecConfig)
    preprocess: PreprocessConfig = Field(default_factory=lambda: PreprocessConfig())
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    general: GeneralConfig = Field(default_factory=GeneralConfig)

    @classmethod
    def from_toml(cls, path: Union[str, Path]) -> "ComictxtConfig":
        p = Path(path)
        with open(p, "rb") as f:
            data = tomllib.load(f)
        return cls(**data)

    def with_overrides(self, overrides: dict[str, Any]) -> "ComictxtConfig":
        """Apply dotted-key overrides, e.g. {'region.conf': 0.35, 'lines.enable_line_stage': False}."""
        data = self.model_dump()
        for dotted, value in overrides.items():
            parts = dotted.split(".")
            node = data
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], dict):
                    raise KeyError(f"Unknown config key: {dotted}")
                node = node[part]
            if parts[-1] not in node:
                raise KeyError(f"Unknown config key: {dotted}")
            node[parts[-1]] = value
        return ComictxtConfig(**data)


def parse_override_value(raw: str) -> Any:
    """Parse a CLI --set value into int/float/bool/list/str."""
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if "," in raw:
        return [parse_override_value(p.strip()) for p in raw.split(",")]
    return raw


def default_config_path() -> Path:
    """Path to the shipped default TOML.

    Prefers the packaged ``comictxt/data/default.toml`` (available when
    installed via pip), falling back to the repo ``configs/default.toml``
    for source checkouts.
    """
    try:
        from importlib.resources import files as _res_files

        res = _res_files("comictxt") / "data" / "default.toml"
        # importlib.resources Traversable may not be a real Path; only use it
        # when it exists on the filesystem.
        res_path = Path(str(res))
        if res_path.is_file():
            return res_path
    except Exception:
        pass
    here = Path(__file__).resolve()
    # comictxt/config.py -> project root / configs/default.toml
    cand = here.parent.parent / "configs" / "default.toml"
    return cand


def user_config_path() -> Path:
    """Editable per-user config location (``~/.config/comictxt/config.toml``).

    Respects ``$XDG_CONFIG_HOME`` and ``$COMICTXT_CONFIG``.
    """
    env = os.environ.get("COMICTXT_CONFIG")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "comictxt" / "config.toml"


def default_config_toml_text() -> str:
    """Return the shipped default TOML text (for ``config --print/--init``)."""
    p = default_config_path()
    try:
        return p.read_text(encoding="utf-8")
    except FileNotFoundError:
        # last resort: dump built-in defaults as TOML-ish via model_dump
        cfg = ComictxtConfig()
        lines = []
        for section, values in cfg.model_dump().items():
            lines.append(f"[{section}]")
            for k, v in values.items():
                if isinstance(v, str):
                    lines.append(f'{k} = "{v}"')
                elif isinstance(v, bool):
                    lines.append(f"{k} = {'true' if v else 'false'}")
                else:
                    lines.append(f"{k} = {v!r}")
            lines.append("")
        return "\n".join(lines)


def resolve_config_file(explicit: Union[str, Path, None]) -> Optional[Path]:
    """Resolve which TOML to load: explicit --config, $COMICTXT_CONFIG, user file."""
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("COMICTXT_CONFIG")
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    u = user_config_path()
    return u if u.is_file() else None


def ensure_user_config() -> Optional[Path]:
    """Create the user config from shipped defaults on first start.

    Returns the path to read (existing or just created), or None when the
    file cannot be written (caller warns and falls back to built-in
    defaults). Never overwrites an existing file: creation is atomic
    (O_CREAT|O_EXCL), so concurrent first starts are safe and user edits
    are never clobbered.
    """
    dest = user_config_path()
    if dest.is_file():
        return dest
    text = (
        "# comictxt user config — auto-generated on first run. Edit freely.\n"
        "# Refresh with new defaults via: comictxt config --init --force\n"
        + default_config_toml_text()
    )
    if not text.endswith("\n"):
        text += "\n"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(dest), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            os.write(fd, text.encode("utf-8"))
        finally:
            os.close(fd)
        return dest
    except FileExistsError:
        # Another process won the first-start race; read their file.
        return dest if dest.is_file() else None
    except OSError:
        return None


def is_offline(cfg: "ComictxtConfig | None" = None) -> bool:
    """True when offline mode is requested via config or environment."""
    if cfg is not None and getattr(getattr(cfg, "general", None), "offline", False):
        return True
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "COMICTXT_OFFLINE"):
        if os.environ.get(var, "").strip().lower() in ("1", "true", "yes"):
            return True
    return False
