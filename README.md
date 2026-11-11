# liveswap

Real-time face swap desktop application for Windows.  
Virtual camera output: Zoom, Meet, Teams, Discord, TikTok Live Studio.

## Two Modes

**Live Mode** — instant, no setup required  
Source face from webcam or photo → real-time swap at 30fps

**Profile Mode** — maximum quality  
Upload 5-10 photos → 15 min personalization → 9.5/10 photorealistic quality

## Requirements

- Windows 10/11
- NVIDIA RTX 2060+ (8GB VRAM recommended)
- Python 3.12

## Stack

- Face detection: InsightFace buffalo_l (SCRFD)
- Swap: AlphaFace_256 / HyperSwap_1b_256
- Enhancement: GPEN-BFR-512
- Parsing: BiSeNet
- Profile: Arc2Face + LoRA fine-tuning
- Virtual camera: softcam (DirectShow, MIT license)
- GUI: PySide6
- Inference: ONNX Runtime FP32 (CUDA, DirectML or CPU). TensorRT FP16 for GPEN is an untested opt-in setting.

## Install

On a clean Windows PC with an NVIDIA card and a current driver, run this one
line in PowerShell. It needs no administrator rights and nothing preinstalled:

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/VladMarkdown/liveswap/main/install.ps1 | iex"
```

It installs everything into `C:\liveswap` (the app, its own Python 3.12, the
packages including the CUDA/cuDNN runtime, and the models) and puts two
shortcuts on the desktop:

- **liveswap** starts the app.
- **liveswap test** runs a 30 second speed test with the webcam and saves
  `liveswap-test-log.txt` to the desktop.

If the install fails, `C:\liveswap\install-log.txt` has the full output.
Running the line again updates the app and keeps the models.

Manual install, with Python 3.12:

```bash
pip install -r requirements.txt
pip install --no-deps "insightface>=2.1"
python download_models.py
python main.py
```

Profile mode additionally needs `requirements-profile.txt`; see the notes in
that file, it does not install cleanly yet.

## Models

`download_models.py` fetches HyperSwap 1b 256, GPEN-BFR-512 and BiSeNet from
the FaceFusion asset releases, and the InsightFace `buffalo_l` pack. Check the
licences of those models before any use beyond testing. AlphaFace_256 has no
public download; if `models/alphaface_256.onnx` is placed by hand it is used
instead of HyperSwap.

The virtual camera is not installed automatically: softcam publishes no
prebuilt binaries, so `softcam.dll` has to be built from source and registered
as administrator. Without it the app still runs and shows its preview.

## Headless benchmark

```bash
python tools/bench.py --photo face.jpg --source clip.mp4
```

Prints fps and per-stage milliseconds as one JSON line per second.
