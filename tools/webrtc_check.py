"""End-to-end check of the phone stream without a phone.

Starts WebRTCStreamer with a synthetic moving picture, then connects to it the
way the browser page does (GET the page, POST an offer, receive video) using
aiortc as the client, and verifies that real, changing frames arrive.

    python tools/webrtc_check.py            # full check, exits 0 on success
    python tools/webrtc_check.py --serve    # only serve; open the printed link on a phone

No camera and no models are used.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faceswap.streaming.webrtc_output import WebRTCStreamer


def _feed(streamer: WebRTCStreamer, stop: threading.Event) -> None:
    """Push a 1280x720 test picture at 30 fps: a counter and a bar that moves."""
    number = 0
    while not stop.is_set():
        frame = np.full((720, 1280, 3), (40, 30, 20), np.uint8)
        x = (number * 12) % 1180
        frame[260:460, x:x + 100] = (60, 200, 250)
        cv2.putText(frame, f"frame {number}", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                    (255, 255, 255), 4, cv2.LINE_AA)
        streamer.send_frame(frame)
        number += 1
        time.sleep(1 / 30)


async def _receive(url: str, wanted: int) -> list[np.ndarray]:
    from aiortc import RTCPeerConnection, RTCSessionDescription

    base, token = url.split("/?k=")
    pc = RTCPeerConnection()
    pc.addTransceiver("video", direction="recvonly")
    got: list[np.ndarray] = []
    done = asyncio.Event()

    @pc.on("track")
    def on_track(track):
        async def pump():
            while len(got) < wanted:
                frame = await track.recv()
                got.append(frame.to_ndarray(format="bgr24"))
            done.set()
        asyncio.ensure_future(pump())

    await pc.setLocalDescription(await pc.createOffer())
    request = urllib.request.Request(
        f"{base}/offer?k={token}",
        data=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}).encode(),
        headers={"Content-Type": "application/json"})
    answer = json.loads(await asyncio.to_thread(lambda: urllib.request.urlopen(request, timeout=10).read()))
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
    try:
        await asyncio.wait_for(done.wait(), timeout=30)
    finally:
        await pc.close()
    return got


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--serve", action="store_true", help="serve until Ctrl+C; test with a real phone")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    streamer = WebRTCStreamer(port=args.port)
    streamer.start()
    url = streamer.local_url()
    stop = threading.Event()
    threading.Thread(target=_feed, args=(streamer, stop), daemon=True).start()
    print("serving", url)
    try:
        if args.serve:
            print("Open the link on a phone on the same Wi-Fi. Ctrl+C to stop.")
            while True:
                time.sleep(1)
                print(f"\rviewers: {streamer.viewers}  ", end="", flush=True)

        page = urllib.request.urlopen(url, timeout=10).read().decode()
        assert "RTCPeerConnection" in page, "page does not contain the player"
        try:
            urllib.request.urlopen(url.split("?")[0], timeout=10)
            raise AssertionError("page served without the access token")
        except urllib.error.HTTPError as error:
            assert error.code == 403, error.code
        print("page: ok, and refused without the token")

        frames = asyncio.run(_receive(url, wanted=45))
        sizes = {f.shape for f in frames}
        tail = frames[15:]                       # skip the encoder's first, coarse frames
        brightness = float(np.mean([f.mean() for f in tail]))
        bar = [int(np.argmax(f[360, :, 1].astype(int) - f[360, :, 0].astype(int))) for f in tail]
        moves = sum(1 for a, b in zip(bar, bar[1:]) if a != b)
        print(f"received {len(frames)} frames, sizes {sizes}, mean brightness {brightness:.1f}, "
              f"bar position changed in {moves} of {len(tail) - 1} steps, viewers seen {streamer.viewers}")
        assert sizes == {(720, 1280, 3)}, sizes
        assert brightness > 20, "frames are black"
        assert moves >= (len(tail) - 1) * 0.6, "picture is not updating"
        print("RESULT: OK, live video reaches a WebRTC client.")
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        stop.set()
        streamer.stop()
        # The port must be free again right away, or the app's next start would fail.
        again = WebRTCStreamer(port=args.port)
        again.start()
        again.stop()
        print("restart after stop: ok")


if __name__ == "__main__":
    raise SystemExit(main())
