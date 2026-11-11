"""Raw ArcFace profile embeddings, using the live detector's shared buffalo_l."""
from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image
from faceswap.core.detector import detect_largest


def extract_embedding_from_image(image_path: str | Path) -> np.ndarray | None:
    """Return the raw 512-dimensional embedding of the largest face."""
    with Image.open(image_path) as image:
        frame = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    face = detect_largest(frame)
    return None if face is None else face.embedding.astype(np.float32)


def build_profile_embedding(image_paths: list[str | Path]) -> np.ndarray:
    """Average raw embeddings; consumers normalize only when their model needs it."""
    embeddings = []
    for path in image_paths:
        emb = extract_embedding_from_image(path)
        if emb is not None:
            embeddings.append(emb)
        else:
            print(f"[profile] no face detected in {path}")
    if not embeddings:
        raise ValueError("No faces detected in any of the provided images")
    return np.stack(embeddings).mean(axis=0).astype(np.float32)


def save_profile(embedding: np.ndarray, save_path: str | Path) -> None:
    np.savez(str(save_path), embedding=embedding)


def load_profile(npz_path: str | Path) -> np.ndarray:
    """Preserve embedding magnitude required by AlphaFace."""
    with np.load(str(npz_path)) as data:
        return data["embedding"].astype(np.float32)
