"""Hayai OCR v2.5 Nova recognizer (PyTorch, @hayaiocr_rec).

Loads JustANormalTinkerer/hayai-ocr-v2.5-nova with trust_remote_code and
exposes a simple ocr_pil() interface. Image preprocessing uses the real
SigLIP2 NaFlex AutoProcessor (resolved from the local HF cache), exactly as
the model card prescribes — the processor pads to max_num_patches with an
attention mask, which the vision encoder and 2D mRoPE require.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image


class TorchHayaiRecognizer:
    """PyTorch HayaiModel (v2.5 Nova) with greedy generation."""

    def __init__(
        self,
        model_path: Union[str, Path],
        processor_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        dtype: str = "float32",
        max_new_tokens: int = 128,
        max_num_patches: int = 384,
        offline: bool = False,
    ) -> None:
        if dtype not in ("float32", "float16"):
            raise ValueError("torch dtype must be 'float32' or 'float16'")
        self.model_path = Path(model_path)
        self.processor_path = Path(processor_path) if processor_path else None
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = int(max_new_tokens)
        self.max_num_patches = int(max_num_patches)
        self.offline = bool(offline)
        self._model = None
        self._tok = None
        self._proc = None
        self._lock = threading.Lock()

    def load(self) -> "TorchHayaiRecognizer":
        import os

        import torch
        from transformers import AutoModel, AutoProcessor, PreTrainedTokenizerFast

        if not (self.model_path / "model.safetensors").exists():
            raise FileNotFoundError(f"torch Hayai weights not found in {self.model_path}")
        # Offline: local snapshot dirs must load without any hub/network access.
        local_only = self.offline or any(
            os.environ.get(v, "").strip().lower() in ("1", "true", "yes")
            for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "COMICTXT_OFFLINE")
        )
        torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32
        try:
            self._model = AutoModel.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                dtype=torch_dtype,
                local_files_only=local_only,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load torch Hayai v2.5-nova model from {self.model_path} "
                f"(offline={local_only}). If you are offline, ensure the snapshot "
                f"dir holds model.safetensors/tokenizer.json/modeling_hayai.py "
                f"or set rec.torch_model explicitly. Original error: {e}"
            ) from e
        self._model.to(self.device).eval()
        try:
            self._tok = PreTrainedTokenizerFast.from_pretrained(
                str(self.model_path), local_files_only=local_only
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Hayai tokenizer from {self.model_path} "
                f"(offline={local_only}). Original error: {e}"
            ) from e
        proc_path = self.processor_path or resolve_rec_processor("")
        try:
            self._proc = AutoProcessor.from_pretrained(
                str(proc_path), local_files_only=local_only
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SigLIP2 processor from {proc_path} "
                f"(offline={local_only}). Set rec.torch_processor explicitly. "
                f"Original error: {e}"
            ) from e
        return self

    def ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    def _preprocess(self, img: Image.Image):
        assert self._proc is not None
        inputs = self._proc(
            images=[img.convert("RGB")],
            max_num_patches=self.max_num_patches,
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(self.device)
        if self.dtype == "float16":
            pixel_values = pixel_values.half()
        attention_mask = inputs["pixel_attention_mask"].to(self.device)
        spatial_shapes = inputs["spatial_shapes"].to(self.device)
        return pixel_values, attention_mask, spatial_shapes

    def ocr_pil(self, img: Image.Image) -> str:
        self.ensure_loaded()
        assert self._model is not None and self._tok is not None
        pixel_values, attention_mask, spatial_shapes = self._preprocess(img)
        with self._lock:
            texts = self._model.generate(
                pixel_values=pixel_values,
                pixel_attention_mask=attention_mask,
                spatial_shapes=spatial_shapes,
                tokenizer=self._tok,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,
                repetition_penalty=1.0,
            )
        text = texts[0] if texts else ""
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        return str(text).strip()

    def ocr_array(self, arr: np.ndarray) -> str:
        if arr.ndim == 2:
            img = Image.fromarray(arr).convert("RGB")
        elif arr.shape[2] == 3:
            img = Image.fromarray(arr).convert("RGB")
        elif arr.shape[2] == 4:
            img = Image.fromarray(arr[:, :, :3]).convert("RGB")
        else:
            raise ValueError(f"Unsupported crop shape: {arr.shape}")
        return self.ocr_pil(img)


def resolve_rec_processor(explicit: str = "") -> Path:
    """Resolve the SigLIP2 NaFlex processor snapshot dir."""
    if explicit:
        p = Path(explicit)
        if not (p / "preprocessor_config.json").exists():
            raise FileNotFoundError(f"SigLIP2 processor not found in: {p}")
        return p
    import os

    roots: list[Path] = []
    env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env:
        roots.append(Path(env))
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    for root in roots:
        snaps = root / "models--google--siglip2-base-patch16-naflex" / "snapshots"
        if not snaps.is_dir():
            continue
        for sha_dir in sorted(snaps.iterdir(), reverse=True):
            if (sha_dir / "preprocessor_config.json").exists():
                return sha_dir
    raise FileNotFoundError(
        "Could not find siglip2-naflex processor in HF cache. Set rec.torch_processor explicitly."
    )


def resolve_rec_torch(explicit: str = "", revision: str = "main") -> Path:
    """Resolve the PyTorch Hayai v2.5-nova snapshot dir for a repo ``revision``.

    Resolution order: explicit path > local snapshot for the revision's
    commit (``refs/<revision>``) > download of that revision (unless
    offline) > newest complete README-bearing snapshot. The mtime
    heuristic alone is not deterministic across repos, so the exact
    revision commit is always preferred when cached.
    """
    if explicit:
        p = Path(explicit)
        if not (p / "model.safetensors").exists():
            raise FileNotFoundError(f"torch Hayai model not found in: {p}")
        return p
    import os

    def complete(d: Path) -> bool:
        return (
            (d / "model.safetensors").exists()
            and (d / "tokenizer.json").exists()
            and (d / "modeling_hayai.py").exists()
        )

    roots: list[Path] = []
    env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env:
        roots.append(Path(env))
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    base = None
    for root in roots:
        cand = root / "models--JustANormalTinkerer--hayai-ocr-v2.5-nova"
        if cand.is_dir():
            base = cand
            break
    if base is None:
        raise FileNotFoundError(
            "Could not find hayai-ocr-v2.5-nova in HF cache. Set rec.torch_model explicitly."
        )

    # exact snapshot for the requested revision, when cached
    if revision:
        ref_file = base / "refs" / revision
        if ref_file.is_file():
            commit = ref_file.read_text(encoding="utf-8").strip()
            snap = base / "snapshots" / commit
            if complete(snap):
                return snap
        # not cached: try to fetch it (no-op offline / already up to date)
        if not os.environ.get("HF_HUB_OFFLINE") and not os.environ.get("TRANSFORMERS_OFFLINE"):
            try:
                from huggingface_hub import snapshot_download

                snap = Path(snapshot_download(
                    "JustANormalTinkerer/hayai-ocr-v2.5-nova", revision=revision))
                if complete(snap):
                    return snap
            except Exception:  # noqa: BLE001 - offline / network failure: fall through
                pass

    snaps = base / "snapshots"
    cands = [d for d in snaps.iterdir() if d.is_dir() and complete(d)] if snaps.is_dir() else []
    if not cands:
        raise FileNotFoundError(
            "No complete hayai-ocr-v2.5-nova snapshot in HF cache. Set rec.torch_model explicitly."
        )
    cands.sort(key=lambda d: ((d / "README.md").exists(), d.stat().st_mtime), reverse=True)
    return cands[0]
