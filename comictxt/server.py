"""owocr-compatible websocket server.

Protocol (must match neokuro/owosocket.py):
    client: send_binary(image_bytes)
    server: send("True")
    server: send(json_string)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

from comictxt.config import ComictxtConfig
from comictxt.pipeline import ComicTxtPipeline

log = logging.getLogger("comictxt.server")


async def _handler(websocket, pipeline: ComicTxtPipeline, pool) -> None:
    async for message in websocket:
        if isinstance(message, str):
            # control / ping messages: acknowledge so clients never hang
            try:
                await websocket.send("True")
            except Exception:
                pass
            continue
        data = bytes(message)
        try:
            await websocket.send("True")
        except Exception:
            continue
        t0 = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(pool, pipeline.process_bytes, data)
            payload = json.dumps(result, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001 - must reply, never hang client
            log.exception("OCR failed: %s", e)
            payload = json.dumps(
                {
                    "image_properties": {"width": 0, "height": 0},
                    "engine_capabilities": {},
                    "paragraphs": [],
                    "error": str(e),
                },
                ensure_ascii=False,
            )
        try:
            await websocket.send(payload)
        except Exception:
            pass
        log.info("served %d bytes in %.2fs", len(data), time.monotonic() - t0)


async def _amain(cfg: ComictxtConfig) -> None:
    import websockets
    from concurrent.futures import ThreadPoolExecutor

    from comictxt.config import resolve_workers

    pipeline = ComicTxtPipeline(cfg, allow_inner_parallel=False)
    log.info("Loading models (this may take a while)...")
    await asyncio.get_running_loop().run_in_executor(None, pipeline.warmup)
    workers = max(1, cfg.server.max_workers or resolve_workers(cfg.general.workers))
    log.info("Models loaded. Listening on %s:%d (%d workers)", cfg.server.host, cfg.server.port, workers)

    async def handler(ws, *args):
        await _handler(ws, pipeline, pool)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="comictxt") as pool:
        async with websockets.serve(
            handler, cfg.server.host, cfg.server.port, max_size=cfg.server.max_size
        ):
            await asyncio.Future()  # run forever


def serve_forever(cfg: ComictxtConfig) -> None:
    import sys

    logging.basicConfig(
        level=getattr(logging, cfg.general.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        asyncio.run(_amain(cfg))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
