"""Train Arc2Face UNet LoRA with projected identity conditioning."""
from __future__ import annotations
from pathlib import Path

from .synthesizer import Arc2FaceSynthesizer


def train_lora(train_data_dir: str | Path, output_dir: str | Path,
               embedding, max_train_steps: int = 800, rank: int = 16,
               learning_rate: float = 1e-4) -> Path:
    """Fine-tune the Arc2Face UNet, never a generic text-conditioned SD UNet."""
    if max_train_steps < 1 or rank < 1:
        raise ValueError("Training steps and LoRA rank must be positive")
    images = sorted(p for p in Path(train_data_dir).iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        raise ValueError("No training images found")
    synthesizer = Arc2FaceSynthesizer()
    try:
        return _train(synthesizer, images, Path(output_dir), embedding,
                      max_train_steps, rank, learning_rate)
    finally:
        synthesizer.close()


def _train(synthesizer, images, output_dir, embedding, steps, rank, learning_rate):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from diffusers import DDPMScheduler, StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    torch.manual_seed(42)
    prompt = synthesizer.prompt_embeddings(embedding).detach()
    pipeline = synthesizer._pipeline
    unet, vae = pipeline.unet, pipeline.vae
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder.to("cpu")
    scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    device, dtype = synthesizer._device, synthesizer._dtype
    latents = []
    with torch.no_grad():
        for path in images:
            with Image.open(path) as image:
                image = ImageOps.fit(image.convert("RGB"), (512, 512))
                pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1
            tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device, dtype)
            latents.append((vae.encode(tensor).latent_dist.sample() * vae.config.scaling_factor).cpu())
    vae.to("cpu")
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                                target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    parameters = [p for p in unet.parameters() if p.requires_grad]
    for parameter in parameters:
        parameter.data = parameter.data.float()
    unet.enable_gradient_checkpointing()
    unet.train()
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    use_amp = device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for step in range(steps):
        latent = latents[step % len(latents)].to(device, dtype)
        noise = torch.randn_like(latent)
        timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device)
        noisy = scheduler.add_noise(latent, noise, timestep)
        # Keep checkpointed blocks differentiable even with a frozen base UNet.
        noisy.requires_grad_(True)
        if scheduler.config.prediction_type == "epsilon":
            target = noise
        elif scheduler.config.prediction_type == "v_prediction":
            target = scheduler.get_velocity(latent, noise, timestep)
        else:
            raise ValueError(f"Unsupported prediction type: {scheduler.config.prediction_type}")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            prediction = unet(noisy, timestep, encoder_hidden_states=prompt).sample
            loss = F.mse_loss(prediction.float(), target.float())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        scaler.step(optimizer)
        scaler.update()
        if (step + 1) % 50 == 0:
            print(f"[trainer] {step + 1}/{steps}: loss={loss.item():.4f}")
    output_dir.mkdir(parents=True, exist_ok=True)
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(
        str(output_dir), unet_lora_layers=state, safe_serialization=True,
    )
    return output_dir / "pytorch_lora_weights.safetensors"
