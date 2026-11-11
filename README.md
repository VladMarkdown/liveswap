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
- Inference: ONNX Runtime FP32 (CUDA, DirectML or CPU). TensorRT is not wired up yet.

## Install

```bash
pip install -r requirements.txt
python main.py
```

## Models

Download models automatically on first launch, or manually place in `models/`.
