"""Hayai OCR v2 ONNX recognizer.

Port of upstream inference_onnx.py (JustANormalTinkerer/hayai-ocr-v2-onnx),
adapted to accept PIL crops and to be driven by ComictxtConfig.
"""
from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

PATCH_SIZE = 16
MAX_PATCHES = 256
MAX_TEXT = 64
D_AXIS = 32
ROPE_THETA = 10000.0

_PRECISION_FILES = {
    "fp32": ("hayai_encoder.onnx", "hayai_decoder.onnx"),
    "fp16": ("hayai_encoder_fp16.onnx", "hayai_decoder_fp16.onnx"),
    "quant": ("hayai_encoder_dynamic_quant.onnx", "hayai_decoder_dynamic_quant.onnx"),
}


def _get_size(h: int, w: int) -> tuple[int, int]:
    EPS = 1e-5

    def scaled(s: float, sz: int) -> int:
        return max(math.ceil((sz * s) / PATCH_SIZE) * PATCH_SIZE, PATCH_SIZE)

    smin, smax = EPS / 10, 100.0
    while smax - smin >= EPS:
        s = (smin + smax) / 2
        th, tw = scaled(s, h), scaled(s, w)
        if (th // PATCH_SIZE) * (tw // PATCH_SIZE) <= MAX_PATCHES:
            smin = s
        else:
            smax = s
    return scaled(smin, h), scaled(smin, w)


def _preprocess_pil(img: Image.Image) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    img = img.convert("RGB")
    w0, h0 = img.size
    th, tw = _get_size(h0, w0)
    ph, pw = th // PATCH_SIZE, tw // PATCH_SIZE
    n = ph * pw
    if img.size != (tw, th):
        img = img.resize((tw, th), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    t = arr.reshape(ph, PATCH_SIZE, pw, PATCH_SIZE, 3).transpose(0, 2, 1, 3, 4).reshape(n, 768)
    pv = np.zeros((1, MAX_PATCHES, 768), dtype=np.float32)
    pv[0, :n] = t
    mask = np.zeros((1, MAX_PATCHES), dtype=np.int64)
    mask[0, :n] = 1
    return pv, mask, (ph, pw)


def _make_mask(L: int) -> np.ndarray:
    mask = np.ones((L, L), dtype=np.float32) * -1e9
    mask[:256, :256] = 0
    mask[:256, 256:] = -1e9
    mask[256:, :256] = 0
    n = L - 256
    if n > 0:
        mask[256:, 256:] = np.where(np.tril(np.ones((n, n), dtype=bool)), 0, -1e9)
    return mask.reshape(1, 1, L, L)


class HayaiRecognizer:
    """ONNX encoder+decoder sessions with host-computed pos-embeds and RoPE."""

    def __init__(
        self,
        onnx_dir: Union[str, Path],
        precision: str = "fp32",
        providers: Optional[list[str]] = None,
        max_new_tokens: int = 128,
    ) -> None:
        if precision not in _PRECISION_FILES:
            raise ValueError(f"precision must be one of {sorted(_PRECISION_FILES)}")
        self.onnx_dir = Path(onnx_dir)
        self.precision = precision
        self.providers = providers or ["CPUExecutionProvider"]
        self.max_new_tokens = int(max_new_tokens)
        self._enc = None
        self._dec = None
        self._tok = None
        self._bos = self._eos = self._pad = None
        self._base: Optional[np.ndarray] = None
        self._pos_cache: dict = {}
        self._rope_cache: dict = {}
        self._lock = threading.Lock()  # guards session run + caches across threads

    # -- loading ---------------------------------------------------------
    def _resolve(self) -> tuple[Path, Path, Path, Path]:
        enc_name, dec_name = _PRECISION_FILES[self.precision]
        enc, dec = self.onnx_dir / enc_name, self.onnx_dir / dec_name
        if not enc.exists() or not dec.exists():
            # fall back to fp32
            enc = self.onnx_dir / "hayai_encoder.onnx"
            dec = self.onnx_dir / "hayai_decoder.onnx"
        base = self.onnx_dir / "position_base.npy"
        if not base.exists():
            base = self.onnx_dir.parent / "position_base.npy"
        tok = self.onnx_dir / "tokenizer.json"
        if not tok.exists():
            tok = self.onnx_dir.parent / "tokenizer.json"
        missing = [str(p) for p in (enc, dec, base, tok) if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"Hayai files missing in {self.onnx_dir}: {missing}")
        return enc, dec, base, tok

    def load(self) -> "HayaiRecognizer":
        import onnxruntime as ort
        from tokenizers import Tokenizer

        enc_path, dec_path, base_path, tok_path = self._resolve()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._enc = ort.InferenceSession(str(enc_path), sess_options=opts, providers=self.providers)
        self._dec = ort.InferenceSession(str(dec_path), sess_options=opts, providers=self.providers)
        self._base = np.load(str(base_path))
        tok = Tokenizer.from_file(str(tok_path))
        self._tok = tok
        self._bos = tok.token_to_id("<bos>")
        self._eos = tok.token_to_id("<eos>")
        self._pad = tok.token_to_id("<pad>")
        return self

    def ensure_loaded(self) -> None:
        if self._enc is None:
            self.load()

    # -- pos / rope (cached) ----------------------------------------------
    def _pos_embeds(self, ph: int, pw: int) -> np.ndarray:
        key = (ph, pw)
        hit = self._pos_cache.get(key)
        if hit is not None:
            return hit
        assert self._base is not None
        base = self._base
        if ph == 16 and pw == 16:
            out = base.reshape(256, 768)
        else:
            base3 = base.reshape(16, 16, 768)
            out = np.empty((ph, pw, 768), dtype=np.float32)
            resample = getattr(Image, "Resampling", Image).BILINEAR
            for c in range(768):
                out[:, :, c] = np.array(
                    Image.fromarray(base3[:, :, c], mode="F").resize((pw, ph), resample),
                    dtype=np.float32,
                )
            out = out.reshape(ph * pw, 768)
        pos = np.zeros((1, MAX_PATCHES, 768), dtype=np.float32)
        pos[0, : ph * pw] = out
        if ph * pw < MAX_PATCHES:
            pos[0, ph * pw :] = out[0:1]
        self._pos_cache[key] = pos
        return pos

    def _rope(self, ph: int, pw: int) -> tuple[np.ndarray, np.ndarray]:
        key = (ph, pw)
        hit = self._rope_cache.get(key)
        if hit is not None:
            return hit
        freqs_half = 1.0 / (ROPE_THETA ** (np.arange(0, D_AXIS, 2, dtype=np.float32) / D_AXIS))
        gy, gx = np.outer(np.arange(ph, dtype=np.float32), freqs_half), np.outer(
            np.arange(pw, dtype=np.float32), freqs_half
        )
        gy = np.expand_dims(gy, 1).repeat(pw, axis=1)
        gx = np.expand_dims(gx, 0).repeat(ph, axis=0)
        vis = np.concatenate([gy, gx], -1).reshape(ph * pw, D_AXIS)
        cos_vis, sin_vis = np.cos(vis), np.sin(vis)
        tf = np.concatenate(
            [np.outer(np.arange(MAX_TEXT, dtype=np.float32), freqs_half)] * 2, -1
        )
        cos_t, sin_t = np.cos(tf), np.sin(tf)
        cos = np.ones((1, 320, D_AXIS), dtype=np.float32)
        sin = np.zeros((1, 320, D_AXIS), dtype=np.float32)
        cos[0, : ph * pw] = cos_vis
        sin[0, : ph * pw] = sin_vis
        cos[0, 256:320] = cos_t
        sin[0, 256:320] = sin_t
        self._rope_cache[key] = (cos, sin)
        return cos, sin

    # -- inference ----------------------------------------------------------
    def ocr_pil(self, img: Image.Image) -> str:
        self.ensure_loaded()
        assert self._enc is not None and self._dec is not None
        pv, mask, (ph, pw) = _preprocess_pil(img)
        with self._lock:
            pos = self._pos_embeds(ph, pw)
            cos_full, sin_full = self._rope(ph, pw)
            visual = self._enc.run(None, {"pixel_values": pv, "attention_mask": mask, "pos_embeds": pos})[0]
            generated = [self._bos]
            for _ in range(self.max_new_tokens):
                L = 256 + len(generated)
                cos, sin = cos_full[:, :L, :], sin_full[:, :L, :]
                logits = self._dec.run(
                    None,
                    {
                        "visual_features": visual,
                        "text_token_ids": np.array([generated], dtype=np.int64),
                        "cos": cos,
                        "sin": sin,
                        "mask": _make_mask(L),
                    },
                )[0]
                nxt = int(np.argmax(logits[0, -1]))
                if nxt in (self._eos, self._pad):
                    break
                generated.append(nxt)
                if len(generated) >= MAX_TEXT:
                    break
            ids = [i for i in generated[1:] if i not in (self._eos, self._pad)]
            return self._tok.decode(ids, skip_special_tokens=True)

    def ocr_array(self, arr: np.ndarray) -> str:
        """arr: HxW[xC] uint8, RGB or BGR or gray."""
        if arr.ndim == 2:
            img = Image.fromarray(arr).convert("RGB")
        elif arr.shape[2] == 3:
            # Heuristic: pipeline passes RGB arrays; accept as-is.
            img = Image.fromarray(arr).convert("RGB")
        elif arr.shape[2] == 4:
            img = Image.fromarray(arr[:, :, :3]).convert("RGB")
        else:
            raise ValueError(f"Unsupported crop shape: {arr.shape}")
        return self.ocr_pil(img)
