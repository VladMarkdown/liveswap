"""Arc2Face identity-conditioned synthesis and session source-face creation."""
from __future__ import annotations
import gc
from pathlib import Path

import numpy as np

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def load_arc2face_component(model_class, subfolder, **kwargs):
    """Try the requested repository spelling, then the upstream spelling."""
    try:
        return model_class.from_pretrained("foivospar/Arc2Face", subfolder=subfolder, **kwargs)
    except OSError:
        return model_class.from_pretrained("FoivosPar/Arc2Face", subfolder=subfolder, **kwargs)


class Arc2FaceSynthesizer:
    def __init__(self, device: str | None = None):
        import torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float16 if self._device.startswith("cuda") else torch.float32
        self._pipeline = None

    def _load_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
        from arc2face import CLIPTextModelWrapper

        encoder = load_arc2face_component(CLIPTextModelWrapper, "encoder", torch_dtype=self._dtype)
        unet = load_arc2face_component(UNet2DConditionModel, "arc2face", torch_dtype=self._dtype)
        pipeline = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL, text_encoder=encoder, unet=unet,
            torch_dtype=self._dtype, safety_checker=None,
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        self._pipeline = pipeline.to(self._device)
        self._pipeline.enable_attention_slicing()

    def prompt_embeddings(self, embedding):
        import torch
        from arc2face import project_face_embs
        self._load_pipeline()
        emb = torch.as_tensor(embedding, dtype=self._dtype, device=self._device).reshape(1, 512)
        norm = emb.norm(dim=1, keepdim=True)
        if not torch.isfinite(emb).all() or norm.item() == 0:
            raise ValueError("Arc2Face requires a finite, nonzero embedding")
        with torch.no_grad():
            return project_face_embs(self._pipeline, emb / norm)

    def generate_images(self, embedding, n_images=1, guidance_scale=3.0, num_inference_steps=25):
        import torch
        prompt_embs = self.prompt_embeddings(embedding)
        with torch.inference_mode():
            return self._pipeline(
                prompt_embeds=prompt_embs, num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale, num_images_per_prompt=n_images,
            ).images

    def generate(self, embedding: np.ndarray, n_images: int = 80,
                 out_dir: str | Path = "train_data", guidance_scale: float = 3.0,
                 num_inference_steps: int = 25) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        generated_paths = []
        for i in range(n_images):
            images = self.generate_images(embedding, 1,
                                          guidance_scale, num_inference_steps)
            for j, image in enumerate(images):
                path = out_dir / f"{i + j:04d}.png"
                image.save(path)
                generated_paths.append(path)
        return generated_paths

    def close(self):
        """Release diffusion models before realtime inference or training starts."""
        import torch
        self._pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_profile_source_face(profile):
    """
    Generate a source face for the live swap session (Fix 3).

    If profile.arc2face_embedding is set (antelopev2 space), it is used directly
    as the Arc2Face conditioning — the correct identity space for the model.
    Otherwise falls back to profile.embedding (buffalo_l) as a proxy.

    Generates up to 5 images, re-embeds each with buffalo_l, and returns the
    face whose embedding has the highest cosine similarity to profile.embedding.
    This is an iterative best-match refinement that works with ONNX models.
    """
    from faceswap.core.detector import detect_largest

    # Choose conditioning: arc2face space (correct) > buffalo_l (approximate proxy)
    arc2face_emb = getattr(profile, 'arc2face_embedding', None)
    cond_embedding = arc2face_emb if arc2face_emb is not None else profile.embedding

    # L2-normalize target embedding for similarity comparison
    tgt_norm = profile.embedding / (np.linalg.norm(profile.embedding) + 1e-9)

    synthesizer = Arc2FaceSynthesizer()
    try:
        synthesizer._load_pipeline()
        if profile.lora_path is not None:
            synthesizer._pipeline.load_lora_weights(str(profile.lora_path))

        best_face = None
        best_sim = -1.0

        for _ in range(5):
            image = synthesizer.generate_images(cond_embedding)[0]
            frame = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
            face = detect_largest(frame)
            if face is None:
                continue
            # Re-embed with buffalo_l; pick the best match to profile.embedding
            sim = float(np.dot(face.normed_embedding, tgt_norm))
            if sim > best_sim:
                best_sim = sim
                best_face = face

        if best_face is None:
            raise ValueError(
                "Arc2Face generated no detectable source face; retry loading the profile"
            )
        return best_face
    finally:
        synthesizer.close()
