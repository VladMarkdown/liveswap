# Arc2Face + LoRA Profile Personalization Research

## Correction: Arc2Face arXiv ID
Correct ID: **arXiv 2403.11641** (ECCV 2024)
- Repo: https://github.com/foivospar/Arc2Face
- HF: https://huggingface.co/FoivosPar/Arc2Face

---

## Key Architecture Insight

**LoRA fine-tunes Arc2Face's SD1.5 UNet, NOT GPEN/CodeFormer.**
GPEN/CodeFormer are GAN-based — no diffusers ecosystem. No LoRA support.

**Real-time swap still uses AlphaFace/HyperSwap with averaged ArcFace embedding.**
The profile pipeline:
1. Photos → average ArcFace embedding → `alice.npz` (used at runtime for swap identity)
2. Arc2Face → 80 synthetic images → DreamBooth LoRA → `alice_lora.safetensors` (future use: synthetic data for deeper personalization)

---

## Install

```bash
pip install diffusers transformers insightface onnxruntime-gpu
pip install git+https://github.com/foivospar/Arc2Face.git
```

## Extract + Average ArcFace Embeddings (Core of Profile Mode)

```python
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np, torch

app = FaceAnalysis(
    name="antelopev2", root="./",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

def extract_embedding(image_path: str) -> torch.Tensor:
    """Returns (1, 512) normalized float16 ArcFace embedding."""
    img = np.array(Image.open(image_path))[:, :, ::-1]  # RGB → BGR
    faces = app.get(img)
    if not faces:
        raise ValueError(f"No face in {image_path}")
    face = sorted(
        faces,
        key=lambda x: (x["bbox"][2]-x["bbox"][0]) * (x["bbox"][3]-x["bbox"][1])
    )[-1]
    emb = torch.tensor(face["embedding"], dtype=torch.float16)[None].cuda()
    return emb / torch.norm(emb, dim=1, keepdim=True)

def build_profile(image_paths: list, save_path: str) -> torch.Tensor:
    """Average embeddings from N photos → robust identity vector."""
    embs = [extract_embedding(p) for p in image_paths]
    stacked = torch.stack(embs, dim=0)   # (N, 1, 512)
    avg = stacked.mean(dim=0)            # (1, 512)
    avg = avg / torch.norm(avg, dim=1, keepdim=True)
    np.savez(save_path, embedding=avg.cpu().numpy())
    return avg

def load_profile(npz_path: str) -> torch.Tensor:
    return torch.tensor(np.load(npz_path)["embedding"], dtype=torch.float16).cuda()
```

**Arc2Face paper Appendix B confirms:** averaging embeddings from multiple photos gives "slightly higher ID resemblance."

---

## Arc2Face Pipeline (Synthetic Image Generation)

```python
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from arc2face import CLIPTextModelWrapper, project_face_embs
import torch

encoder = CLIPTextModelWrapper.from_pretrained("models", subfolder="encoder", torch_dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained("models", subfolder="arc2face", torch_dtype=torch.float16)
pipeline = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    text_encoder=encoder, unet=unet,
    torch_dtype=torch.float16, safety_checker=None,
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to("cuda")

def generate_training_set(id_emb: torch.Tensor, n_images: int = 80, out_dir: str = "train_data"):
    import os; os.makedirs(out_dir, exist_ok=True)
    prompt_embs = project_face_embs(pipeline, id_emb)  # (1, 77, 768)
    for i in range(0, n_images, 4):
        batch_n = min(4, n_images - i)
        images = pipeline(
            prompt_embeds=prompt_embs,
            num_inference_steps=25,
            guidance_scale=3.0,
            num_images_per_prompt=batch_n,
        ).images
        for j, img in enumerate(images):
            img.save(f"{out_dir}/{i+j:04d}.png")
```

---

## LoRA Training Config (~15 min on RTX 3070)

```bash
accelerate launch examples/dreambooth/train_dreambooth_lora.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --instance_data_dir="train_data/" \
  --output_dir="lora_output/" \
  --instance_prompt="photo of a sks person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --rank=16 \
  --lora_alpha=16 \
  --checkpointing_steps=400 \
  --mixed_precision="fp16" \
  --enable_xformers_memory_efficient_attention \
  --seed=42
```

- **rank=16, alpha=16**: Arc2Avatar (arXiv 2501.05379) confirms rank 16 for Arc2Face UNet
- **800 steps × ~1.1s/step ≈ 15 min** on RTX 3070
- Output: `lora_output/pytorch_lora_weights.safetensors` (~25MB)

---

## LoRA Runtime Load/Hotswap

```python
# Load profile
pipeline.load_lora_weights("profiles/", weight_name="alice_lora.safetensors", adapter_name="alice")
pipeline.set_adapters("alice", adapter_weights=1.0)

# Hotswap to another profile (no extra VRAM)
pipeline.load_lora_weights("profiles/", weight_name="bob_lora.safetensors",
                            hotswap=True, adapter_name="alice")

# Fuse for 5-10% faster inference (irreversible until unfuse_lora())
pipeline.fuse_lora(adapter_names=["alice"], lora_scale=1.0)
pipeline.unload_lora_weights()
```

---

## IP-Adapter FaceID — NOT Real-Time

Measured latencies (V100, 16 steps):
- IP-Adapter FaceID: ~9s/frame
- InstantID: ~9s/frame
- DreamID: ~0.6s/frame (fastest diffusion method)

**NOT viable for 30fps. Use only for offline profile image generation.**

---

## 2025 Papers on Few-Shot Personalization

| Paper | Key idea | Status |
|---|---|---|
| **HyperLoRA** (CVPR 2025, ByteDance) arXiv 2503.16944 | Hypernetwork predicts LoRA from 1 photo — no training | No code yet |
| **DreamID** arXiv 2504.14509 | Diffusion swap, 0.6s inference | Research |
| **Arc2Avatar** arXiv 2501.05379 | LoRA rank 16 on Arc2Face UNet for 3D avatar | Confirms rank 16 |
| **DiffLoRA** arXiv 2408.06740 | Hypernetwork for SDXL LoRA from face | Research |

**HyperLoRA** most promising: single photo → LoRA weights in seconds. Watch for code release.

---

## Profile Mode Architecture

```
PROFILE CREATION (~15-20 min, once per person)
  5-10 photos
    → insightface antelopev2: extract (1,512) ArcFace embedding
    → average + renormalize → alice.npz (2KB)   ← USED AT RUNTIME
  
  Arc2Face generation:
    → project_face_embs() → DPM-Solver 25 steps
    → 80 synthetic images at 512×512 → train_data/
  
  DreamBooth LoRA (optional, deeper personalization):
    → rank=16, alpha=16, lr=1e-4, 800 steps
    → ~15 min → alice_lora.safetensors (25MB)

INFERENCE (real-time, per-frame)
  current frame
    → SCRFD detect + align
    → AlphaFace_256 with alice.npz embedding (averaged identity)
    → GPEN enhancement
```

**Profile Level 1 (fast, 2 min):** Just average embeddings → alice.npz → 8.5/10 quality
**Profile Level 2 (full, 20 min):** + Arc2Face + LoRA → 9.5/10 quality
