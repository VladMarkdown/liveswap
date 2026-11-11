"""
WebRTC phone output via aiortc + aiohttp.

Starts a local HTTP server on port 8765. Opening http://localhost:8765 on the
phone's browser shows the live swapped face stream via WebRTC.

Usage:
    streamer = WebRTCStreamer()
    streamer.start()
    # ... in the frame loop:
    streamer.send_frame(bgr_array)
    # ... on close:
    streamer.stop()

Requires:  pip install aiortc aiohttp

The HTML page served handles the WebRTC offer/answer handshake automatically.
"""
from __future__ import annotations
import asyncio
import gc
import logging
import queue
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_PORT = 8765
_ICE_SERVERS = []  # local LAN only — no STUN/TURN needed

# Minimal HTML page served to the phone browser
_INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>liveswap</title>
<style>
body { margin:0; background:#111; display:flex; align-items:center;
       justify-content:center; height:100vh; }
video { max-width:100%; max-height:100vh; }
#status { color:#aaa; font-family:sans-serif; font-size:14px; text-align:center; }
</style>
</head>
<body>
<div>
  <video id="v" autoplay playsinline muted></video>
  <p id="status">Connecting…</p>
</div>
<script>
async function start() {
  const pc = new RTCPeerConnection({iceServers: []});
  pc.ontrack = e => {
    document.getElementById('v').srcObject = e.streams[0];
    document.getElementById('status').textContent = 'Live';
  };
  pc.oniceconnectionstatechange = () => {
    if (pc.iceConnectionState === 'disconnected') {
      document.getElementById('status').textContent = 'Disconnected — reload to reconnect';
    }
  };
  pc.addTransceiver('video', {direction: 'recvonly'});
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  const r = await fetch('/offer', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({sdp: offer.sdp, type: offer.type})
  });
  const answer = await r.json();
  await pc.setRemoteDescription(answer);
}
start();
</script>
</body>
</html>
"""


class _LiveVideoTrack:
    """aiortc VideoStreamTrack that sources frames from a thread-safe queue."""

    def __init__(self):
        try:
            from aiortc import VideoStreamTrack
            from av import VideoFrame
            self._VideoFrame = VideoFrame
            self._base_cls = VideoStreamTrack
        except ImportError:
            raise ImportError("aiortc / av not installed: pip install aiortc")

        class _Track(self._base_cls):
            kind = "video"
            _frame_q: queue.Queue

            async def recv(self):
                try:
                    bgr = self._frame_q.get_nowait()
                except queue.Empty:
                    bgr = None
                if bgr is None:
                    # Return last or blank frame
                    rgb = np.zeros((720, 1280, 3), np.uint8)
                else:
                    rgb = bgr[:, :, ::-1]
                frame = self._frame_cls.from_ndarray(rgb, format="rgb24")
                pts, timebase = await self._next_pts()
                frame.pts = pts
                frame.time_base = timebase
                return frame

        self._TrackCls = _Track
        self._track = None
        self._frame_q: queue.Queue = queue.Queue(maxsize=1)

    def build(self):
        from av import VideoFrame
        track = self._TrackCls()
        track._frame_q = self._frame_q
        track._frame_cls = VideoFrame
        # Monkey-patch _next_pts from the base class method
        async def _next_pts(self_inner=track):
            pts = getattr(self_inner, '_pts', 0)
            self_inner._pts = pts + 1
            from fractions import Fraction
            return pts, Fraction(1, 30)
        track._next_pts = _next_pts
        self._track = track
        return track

    def push(self, bgr: np.ndarray) -> None:
        try:
            self._frame_q.put_nowait(bgr)
        except queue.Full:
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_q.put_nowait(bgr)
            except queue.Full:
                pass


class WebRTCStreamer:
    """
    Starts a local HTTP server on port 8765 and streams the swapped face
    to any browser that opens http://<local-ip>:8765.
    """

    def __init__(self, port: int = _PORT):
        self._port = port
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._site = None
        self._track: Optional[_LiveVideoTrack] = None
        self._active = False

    def start(self) -> None:
        if self._active:
            return
        self._track = _LiveVideoTrack()
        self._active = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="webrtc")
        self._thread.start()

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception:
            logger.exception("WebRTC server error")
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        try:
            from aiohttp import web
            from aiortc import RTCPeerConnection, RTCSessionDescription
            import json
        except ImportError as e:
            logger.error("WebRTC deps missing: %s — pip install aiortc aiohttp", e)
            return

        pcs: set = set()

        async def _index(request):
            return web.Response(text=_INDEX_HTML, content_type="text/html")

        async def _offer(request):
            params = await request.json()
            offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
            pc = RTCPeerConnection()
            pcs.add(pc)

            @pc.on("connectionstatechange")
            async def on_state():
                if pc.connectionState in ("failed", "closed"):
                    await pc.close()
                    pcs.discard(pc)

            video_track = self._track.build()
            pc.addTrack(video_track)
            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            return web.Response(
                content_type="application/json",
                text=json.dumps({"sdp": pc.localDescription.sdp,
                                 "type": pc.localDescription.type}),
            )

        app = web.Application()
        app.router.add_get("/", _index)
        app.router.add_post("/offer", _offer)
        runner = web.AppRunner(app)
        await runner.setup()
        self._site = web.TCPSite(runner, "0.0.0.0", self._port)
        await self._site.start()
        logger.info("WebRTC server listening on http://0.0.0.0:%d", self._port)

        # Keep running until stopped
        while self._active:
            await asyncio.sleep(1)

        await runner.cleanup()
        for pc in list(pcs):
            await pc.close()

    def send_frame(self, bgr_array: np.ndarray) -> None:
        """Push a BGR frame to all connected WebRTC clients."""
        if self._track is not None and self._active:
            self._track.push(bgr_array)

    def is_active(self) -> bool:
        return self._active

    def stop(self) -> None:
        self._active = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._thread = None
        self._loop = None
        gc.collect()

    def local_url(self) -> str:
        """Return the URL to display / generate QR code for."""
        import socket
        hostname = socket.gethostname()
        try:
            ip = socket.gethostbyname(hostname)
        except Exception:
            ip = "127.0.0.1"
        return f"http://{ip}:{self._port}"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
