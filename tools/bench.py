"""Headless pipeline run with the real models: no GUI, no virtual camera.

Prints one JSON line per second with fps and per-stage milliseconds, so a run
on another machine can be watched live over SSH:

    ssh gpu-pc "cd liveswap && python tools/bench.py --photo face.jpg --source clip.mp4" | tee run.jsonl

Model providers (CUDA / CPU fallback) and errors go to stderr as normal log lines.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--photo", required=True, help="source face image")
    parser.add_argument("--source", default="0",
                        help="video file, or a camera index (default 0)")
    parser.add_argument("--seconds", type=float, default=30.0, help="how long to run")
    parser.add_argument("--quality", choices=("fast", "ultra"), default="ultra")
    parser.add_argument("--pixel-boost", action="store_true")
    parser.add_argument("--tensorrt", action="store_true", help="TensorRT FP16 for GPEN")
    parser.add_argument("--models", default="models", help="folder with the ONNX models")
    parser.add_argument("--save", help="write the last output frame to this image file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(name)s %(message)s")

    from faceswap.core.detector import detect_largest
    from faceswap.core.enhancer import FaceEnhancer
    from faceswap.core.parser import FaceParser
    from faceswap.core.pipeline import SwapPipeline, fit_frame
    from faceswap.core.swapper import load_default_swapper

    # np.fromfile + imdecode reads non-ASCII paths, which cv2.imread cannot on Windows.
    try:
        photo = cv2.imdecode(np.fromfile(args.photo, dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        photo = None
    if photo is None:
        parser.error(f"cannot read {args.photo}")
    source_face = detect_largest(photo)
    if source_face is None:
        parser.error("no face found in --photo")

    models = Path(args.models)
    pipeline = SwapPipeline(
        load_default_swapper(models),
        FaceEnhancer(models / "GPEN-BFR-512.onnx", tensorrt=args.tensorrt),
        FaceParser(models / "bisenet.onnx"),
        quality=args.quality, pixel_boost=args.pixel_boost)
    pipeline.set_source_face(source_face)

    is_camera = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_camera else args.source)
    if not cap.isOpened():
        parser.error(f"cannot open {args.source}")

    started = last_report = time.perf_counter()
    frames = reported = 0
    output = None
    while time.perf_counter() - started < args.seconds:
        ok, frame = cap.read()
        if not ok:
            if is_camera:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop the clip
            continue
        output = pipeline.process_frame(fit_frame(frame, 1280, 720))
        frames += 1
        now = time.perf_counter()
        if now - last_report >= 1.0:
            print(json.dumps({
                "t": round(now - started, 1),
                "fps": round((frames - reported) / (now - last_report), 1),
                "frame_ms": round(pipeline.average_inference_ms, 1),
                "gpen_skipped": pipeline.gpen_skipped,
                "stages_ms": {k: round(v, 1) for k, v in pipeline.stage_ms.items()},
            }), flush=True)
            last_report, reported = now, frames
    cap.release()

    if args.save and output is not None:
        cv2.imencode(Path(args.save).suffix or ".png", output)[1].tofile(args.save)
    print(json.dumps({"done": True, "frames": frames,
                      "mean_fps": round(frames / max(time.perf_counter() - started, 1e-6), 1)}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
