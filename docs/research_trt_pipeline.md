# TRT FP16 / ONNX Runtime Pipeline Research

## Critical: FP16 Precision Issues

| Model | FP16 safe | Notes |
|---|---|---|
| SCRFD | YES | detection robust |
| GPEN-BFR-512 | YES | GAN tested in FP16 |
| **AlphaFace_256 / inswapper** | **NO** | garbled faces, FP16 overflow in normalization. ORT issue #19437 |
| **BiSeNet** | **PROBABLY NO** | Softmax over many classes fragile in FP16 |

For unsafe models, use CUDA FP32:
```python
fp32_prov = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
sess_swap    = ort.InferenceSession("alphaface_256.onnx", providers=fp32_prov)
sess_bisenet = ort.InferenceSession("bisenet.onnx",       providers=fp32_prov)
```

## TensorRT EP Full Config

```python
providers = [
    (
        "TensorrtExecutionProvider",
        {
            "trt_fp16_enable": True,
            "trt_int8_enable": False,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "./trt_cache",
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": "./trt_cache",
            "trt_builder_optimization_level": 3,
            "trt_max_workspace_size": 6 * (1 << 30),   # 6 GB
            "trt_profile_min_shapes": "input:1x3x128x128",
            "trt_profile_max_shapes": "input:1x3x512x512",
            "trt_profile_opt_shapes": "input:1x3x256x256",
            "trt_auxiliary_streams": -1,
            "device_id": 0,
        },
    ),
    (
        "CUDAExecutionProvider",
        {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": 6 * (1 << 30),
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        },
    ),
    "CPUExecutionProvider",
]
```

## Expected Latency on RTX 3070

| Stage | EP | ms |
|---|---|---|
| SCRFD 640px | TRT FP16 | 3–5 |
| AlphaFace_256 | CUDA FP32 | 8–12 |
| BiSeNet | CUDA FP32 | 5–8 |
| GPEN-BFR-512 | TRT FP16 | 15–25 ← bottleneck |
| Color/blend CPU | — | 1–3 |
| **Total sequential** | | **32–53 ms** |

**Strategies to hit 30fps:**
- Skip GPEN every 2nd frame (barely noticeable)
- Run GPEN at 256px input + upsample (~4x faster)
- Use `trt_auxiliary_streams: 2` to overlap GPEN with next SCRFD
- Double-buffer pipeline hides latency behind capture

## GPU Auto-Detect + Provider Selection

```python
import subprocess
import onnxruntime as ort

def detect_gpu_vendor() -> str:
    try:
        name = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "name"],
            capture_output=True, text=True, timeout=5
        ).stdout.upper()
        if "NVIDIA" in name: return "nvidia"
        if "AMD" in name or "RADEON" in name: return "amd"
    except Exception:
        pass
    avail = ort.get_available_providers()
    if "TensorrtExecutionProvider" in avail or "CUDAExecutionProvider" in avail:
        return "nvidia"
    if "DmlExecutionProvider" in avail:
        return "amd"
    return "cpu"

def build_providers(vendor: str, trt_cache: str = "./trt_cache") -> list:
    if vendor == "nvidia":
        return [
            ("TensorrtExecutionProvider", {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": trt_cache,
                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": trt_cache,
                "trt_builder_optimization_level": 3,
                "trt_max_workspace_size": 6 * (1 << 30),
                "trt_auxiliary_streams": -1,
                "device_id": 0,
            }),
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
    if vendor == "amd":
        return [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
```

## Async 4-Thread Double-Buffer Pipeline

```python
import threading, queue, numpy as np

RAW_Q  = queue.Queue(maxsize=2)
PRE_Q  = queue.Queue(maxsize=2)
POST_Q = queue.Queue(maxsize=2)
STOP   = object()

def capture_thread(cap):
    while True:
        ret, frame = cap.read()
        if not ret: break
        try: RAW_Q.put_nowait(frame)
        except queue.Full: pass
    RAW_Q.put(STOP)

def preprocess_thread():
    while True:
        item = RAW_Q.get()
        if item is STOP:
            PRE_Q.put(STOP); return
        blob = preprocess(item)  # CPU: resize, normalize → float32 NCHW
        try: PRE_Q.put_nowait((item, blob))
        except queue.Full: pass

def infer_thread(sessions):
    # ort.run() releases GIL during CUDA kernels
    while True:
        item = PRE_Q.get()
        if item is STOP:
            POST_Q.put(STOP); return
        orig, blob = item
        result = run_pipeline(sessions, blob, orig)
        try: POST_Q.put_nowait(result)
        except queue.Full: pass

def display_thread(callback):
    while True:
        frame = POST_Q.get()
        if frame is STOP: return
        callback(frame)
```

## IOBinding (Zero-Copy GPU Transfer)

```python
import onnxruntime as ort

ort_in = ort.OrtValue.ortvalue_from_numpy(blob, "cuda", 0)
io = session.io_binding()
io.bind_input(name="input", device_type="cuda", device_id=0,
              element_type=np.float32, shape=blob.shape,
              buffer_ptr=ort_in.data_ptr())
io.bind_output(name="output", device_type="cuda", device_id=0)
session.run_with_iobinding(io)
result_np = io.get_outputs()[0].numpy()  # single D2H only here
```

## TRT First-Run Build Times (RTX 3070)

| Model | First build | Cache hit |
|---|---|---|
| SCRFD | ~20–60s | <1s |
| AlphaFace_256 | ~40–90s | <1s |
| BiSeNet | ~20–60s | <1s |
| GPEN-BFR-512 | **3–8 min** | <1s |

Progress ticker while building:
```python
import threading, time, onnxruntime as ort

def create_session(model_path: str, providers: list, label: str = ""):
    done = threading.Event()
    t0 = time.monotonic()

    def ticker():
        while not done.is_set():
            print(f"\r  Building TRT engine [{label}] ... {time.monotonic()-t0:.0f}s",
                  end="", flush=True)
            done.wait(5)
        print()

    t = threading.Thread(target=ticker, daemon=True)
    t.start()
    try:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    finally:
        done.set(); t.join()
    return sess
```

## Install

```bash
# NVIDIA (TRT + CUDA)
pip install onnxruntime-gpu==1.21.0
# needs: CUDA 12.x + cuDNN 8.9.7 + TensorRT 10.x

# AMD / Intel Arc
pip install onnxruntime-directml==1.21.0
```
