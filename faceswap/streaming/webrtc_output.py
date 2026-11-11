"""
WebRTC output to a phone (or any) browser on the same network, via aiortc + aiohttp.

    streamer = WebRTCStreamer()
    streamer.start()                 # raises if the server cannot start
    url = streamer.local_url()       # http://<lan-ip>:8765/?k=<token>, for a QR code
    streamer.send_frame(bgr_array)   # from the frame loop, any thread
    streamer.stop()

The page at that URL makes a receive-only WebRTC connection back to this
server and plays the video. Signalling is one HTTP POST; media goes directly
between the two devices over UDP, so they must be able to reach each other
(same Wi-Fi, no client isolation) and the PC's firewall must allow Python.

Requires:  pip install aiortc aiohttp
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import secrets
import socket
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_PORT = 8765

# Page served to the browser. __TOKEN__ is replaced with the access token.
_INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>liveswap</title>
<style>
html, body { margin:0; height:100%; background:#000; }
body { display:flex; flex-direction:column; align-items:center; justify-content:center; }
video { max-width:100%; max-height:100%; }
#status { position:fixed; left:0; right:0; bottom:12px; color:#aaa;
          font-family:sans-serif; font-size:14px; text-align:center; }
</style>
</head>
<body>
<video id="v" autoplay playsinline muted></video>
<p id="status">Connecting…</p>
<script>
const video = document.getElementById('v');
const statusLine = document.getElementById('status');
let pc = null;
let retry = null;

function say(text) { statusLine.textContent = text; }

function reconnectSoon() {
  if (retry) return;
  retry = setTimeout(() => { retry = null; start(); }, 2000);
}

// aiortc does not trickle candidates, so send the offer once gathering is done.
function gatheringDone(conn) {
  return new Promise(resolve => {
    if (conn.iceGatheringState === 'complete') return resolve();
    const timer = setTimeout(resolve, 2000);
    conn.addEventListener('icegatheringstatechange', () => {
      if (conn.iceGatheringState === 'complete') { clearTimeout(timer); resolve(); }
    });
  });
}

async function start() {
  if (pc) { pc.close(); }
  say('Connecting…');
  const conn = pc = new RTCPeerConnection({iceServers: []});
  conn.ontrack = e => {
    video.srcObject = e.streams[0] || new MediaStream([e.track]);
    video.play().catch(() => {});
  };
  conn.onconnectionstatechange = () => {
    if (conn !== pc) return;
    if (conn.connectionState === 'connected') say('');
    if (conn.connectionState === 'failed' || conn.connectionState === 'disconnected') {
      say('Connection lost. Reconnecting…');
      reconnectSoon();
    }
  };
  try {
    conn.addTransceiver('video', {direction: 'recvonly'});
    await conn.setLocalDescription(await conn.createOffer());
    await gatheringDone(conn);
    const reply = await fetch('/offer?k=__TOKEN__', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({sdp: conn.localDescription.sdp, type: conn.localDescription.type})
    });
    if (!reply.ok) throw new Error('server said ' + reply.status);
    await conn.setRemoteDescription(await reply.json());
  } catch (err) {
    say('Cannot connect (' + err.message + '). Retrying…');
    reconnectSoon();
  }
}
start();
</script>
</body>
</html>
"""


def _make_track(streamer: "WebRTCStreamer"):
    """A video track that sends the streamer's latest frame at a steady 30 fps."""
    from aiortc import VideoStreamTrack
    from av import VideoFrame

    class LatestFrameTrack(VideoStreamTrack):
        async def recv(self):
            # next_timestamp() sleeps to keep 30 fps; without it the sender
            # spins and floods the connection.
            pts, time_base = await self.next_timestamp()
            bgr = streamer._latest
            if bgr is None:
                bgr = streamer._blank
            frame = VideoFrame.from_ndarray(np.ascontiguousarray(bgr), format="bgr24")
            frame.pts = pts
            frame.time_base = time_base
            return frame

    return LatestFrameTrack()


def lan_address() -> str:
    """The address other devices on the local network reach this PC at.

    Asks the OS which interface it would use for an outside address; no packet
    is sent. gethostbyname(hostname) often returns a virtual adapter instead.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))      # TEST-NET-1: never routed, never contacted
        return probe.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        probe.close()


class WebRTCStreamer:
    """Serves a page on the local network that plays the frames given to send_frame()."""

    def __init__(self, port: int = _PORT):
        self._port = port
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self._active = False
        # Every viewer's track reads this; a queue would hand each frame to one viewer only.
        self._latest: Optional[np.ndarray] = None
        self._blank = np.zeros((720, 1280, 3), np.uint8)
        # Without it anyone on the network who guesses the port could watch.
        self._token = secrets.token_urlsafe(9)
        self.viewers = 0

    def start(self, timeout: float = 10.0) -> None:
        """Start the server and wait until it is listening. Raises if it cannot start."""
        if self._active:
            return
        self._ready.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="webrtc")
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("WebRTC server did not start in time")
        if self._error is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
            raise RuntimeError(f"WebRTC server failed to start: {self._error}") from self._error
        self._active = True

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except BaseException as exc:      # reported to start() through _error
            logger.exception("WebRTC server error")
            self._error = exc
        finally:
            self._ready.set()
            self._loop.close()

    async def _serve(self) -> None:
        from aiohttp import web
        from aiortc import RTCPeerConnection, RTCSessionDescription

        self._stop_event = asyncio.Event()
        pcs: set = set()

        def allowed(request) -> bool:
            return hmac.compare_digest(request.query.get("k", ""), self._token)

        async def index(request):
            if not allowed(request):
                return web.Response(status=403, text="Open the link from the QR code.")
            return web.Response(text=_INDEX_HTML.replace("__TOKEN__", self._token),
                                content_type="text/html")

        async def offer(request):
            if not allowed(request):
                return web.Response(status=403, text="forbidden")
            params = await request.json()
            pc = RTCPeerConnection()
            pcs.add(pc)

            @pc.on("connectionstatechange")
            async def on_state():
                logger.info("WebRTC viewer %s", pc.connectionState)
                if pc.connectionState in ("failed", "closed"):
                    await pc.close()
                    pcs.discard(pc)
                self.viewers = sum(1 for p in pcs if p.connectionState == "connected")

            pc.addTrack(_make_track(self))
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=params["sdp"], type=params["type"]))
            await pc.setLocalDescription(await pc.createAnswer())
            return web.json_response({"sdp": pc.localDescription.sdp,
                                      "type": pc.localDescription.type})

        app = web.Application()
        app.router.add_get("/", index)
        app.router.add_post("/offer", offer)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", self._port).start()
        logger.info("WebRTC server listening on port %d", self._port)
        self._ready.set()

        await self._stop_event.wait()

        await asyncio.gather(*(pc.close() for pc in list(pcs)), return_exceptions=True)
        await runner.cleanup()
        self.viewers = 0

    def send_frame(self, bgr_array: np.ndarray) -> None:
        """Make this BGR frame the one every connected viewer receives next."""
        if self._active:
            self._latest = bgr_array

    def is_active(self) -> bool:
        return self._active

    def stop(self) -> None:
        self._active = False
        loop, stop_event = self._loop, self._stop_event
        if loop is not None and stop_event is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(stop_event.set)
            except RuntimeError:
                pass                       # the loop closed between the check and the call
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None
        self._loop = None
        self._stop_event = None
        self._latest = None

    def local_url(self) -> str:
        """The link to open on the phone; put this in the QR code."""
        return f"http://{lan_address()}:{self._port}/?k={self._token}"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
