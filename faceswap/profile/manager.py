"""
Profile management: create, save, load, list profiles.

Profile files:
  profiles/<name>/<name>.npz          - ArcFace embedding (2KB)
  profiles/<name>/<name>_lora.safetensors - LoRA weights (25MB, optional)

Level 1 profile: .npz only → 8.5/10 quality, ~2 min to create
Level 2 profile: .npz + .safetensors → 9.5/10 quality, ~20 min to create
"""
from __future__ import annotations
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .extractor import build_profile_embedding, save_profile, load_profile
from .synthesizer import Arc2FaceSynthesizer, generate_profile_source_face
from .trainer import train_lora

PROFILES_DIR = Path("profiles")


@dataclass
class Profile:
    name: str
    embedding: np.ndarray  # (512,) float32
    lora_path: Path | None = None  # Level 2 only
    level: int = 1  # 1 or 2


def list_profiles() -> list[str]:
    """Return list of profile names."""
    if not PROFILES_DIR.exists():
        return []
    return [d.name for d in PROFILES_DIR.iterdir() if d.is_dir()]


def load_profile_by_name(name: str) -> Profile:
    """Load a profile by name."""
    profile_dir = PROFILES_DIR / name
    npz_path = profile_dir / f"{name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Profile '{name}' not found")

    embedding = load_profile(npz_path)
    lora_path = profile_dir / f"{name}_lora.safetensors"
    level = 2 if lora_path.exists() else 1

    return Profile(
        name=name,
        embedding=embedding,
        lora_path=lora_path if level == 2 else None,
        level=level,
    )


def delete_profile(name: str) -> None:
    profile_dir = PROFILES_DIR / name
    if profile_dir.exists():
        shutil.rmtree(profile_dir)


def create_profile_level1(
    name: str,
    image_paths: list[str | Path],
    progress_callback: Callable[[str], None] | None = None,
) -> Profile:
    """
    Level 1 profile: extract and average ArcFace embeddings.
    ~2 min. Quality: 8.5/10.
    """
    def log(msg):
        print(f"[profile] {msg}")
        if progress_callback:
            progress_callback(msg)

    log(f"Creating Level 1 profile '{name}' from {len(image_paths)} photos...")

    profile_dir = PROFILES_DIR / name
    profile_dir.mkdir(parents=True, exist_ok=True)

    log("Extracting ArcFace embeddings...")
    embedding = build_profile_embedding(image_paths)

    npz_path = profile_dir / f"{name}.npz"
    save_profile(embedding, npz_path)
    log(f"Saved → {npz_path}")

    return Profile(name=name, embedding=embedding, level=1)


def create_profile_level2(
    name: str,
    image_paths: list[str | Path],
    n_synthetic: int = 80,
    progress_callback: Callable[[str], None] | None = None,
) -> Profile:
    """
    Level 2 profile: embedding + Arc2Face synthesis + LoRA training.
    ~20 min on RTX 3070. Quality: 9.5/10.
    """
    def log(msg):
        print(f"[profile] {msg}")
        if progress_callback:
            progress_callback(msg)

    profile_dir = PROFILES_DIR / name
    profile_dir.mkdir(parents=True, exist_ok=True)

    log(f"Creating Level 2 profile '{name}'...")

    # Step 1: Extract embedding
    log("Step 1/3: Extracting ArcFace embeddings...")
    embedding = build_profile_embedding(image_paths)
    npz_path = profile_dir / f"{name}.npz"
    save_profile(embedding, npz_path)

    # Step 2: Generate synthetic images
    log(f"Step 2/3: Generating {n_synthetic} synthetic images via Arc2Face...")
    train_dir = profile_dir / "train_data"
    synthesizer = Arc2FaceSynthesizer()
    try:
        synthesizer.generate(embedding, n_images=n_synthetic, out_dir=train_dir)
    finally:
        synthesizer.close()

    # Step 3: LoRA training
    log("Step 3/3: LoRA fine-tuning (~15 min)...")
    lora_output_dir = profile_dir / "lora_output"
    lora_path_tmp = train_lora(
        train_data_dir=train_dir,
        output_dir=lora_output_dir,
        embedding=embedding,
    )

    # Move LoRA weights to profile dir
    lora_path = profile_dir / f"{name}_lora.safetensors"
    shutil.copy(lora_path_tmp, lora_path)

    log(f"Profile ready → {profile_dir}")
    return Profile(name=name, embedding=embedding, lora_path=lora_path, level=2)
