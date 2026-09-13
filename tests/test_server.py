"""Websocket integration test with a stub pipeline (no weights)."""
import asyncio
import json

import websockets

from comictxt.config import ComictxtConfig
from comictxt.server import _handler


class StubPipeline:
    def process_bytes(self, data: bytes) -> dict:
        assert len(data) > 0
        return {
            "image_properties": {"width": 10, "height": 10},
            "engine_capabilities": {},
            "paragraphs": [],
        }


def test_owocr_handshake_ack_then_json():
    async def run():
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (255, 0, 0)).save(buf, format="PNG")
        payload = buf.getvalue()

        async def handler(ws):
            await _handler(ws, StubPipeline(), None)

        async with websockets.serve(handler, "127.0.0.1", 18731, max_size=10_000_000):
            async with websockets.connect("ws://127.0.0.1:18731") as ws:
                await ws.send(payload)  # binary, like neokuro OwocrWebsocket
                ack = await ws.recv()
                assert ack == "True"
                resp = await ws.recv()
                data = json.loads(resp)
                assert data["image_properties"] == {"width": 10, "height": 10}
                assert "paragraphs" in data

    asyncio.run(run())
