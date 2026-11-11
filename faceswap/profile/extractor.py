"""
ArcFace profile embedding extraction with outlier trimming and dual-space support.

buffalo_l (InsightFace) embeddings: for the AlphaFace/inswapper face swap model.
antelopev2 / arc2face_arcface embeddings: for Arc2Face identity conditioning.
"""
from __future__ import annotations
from pathlib import Path
from threading import RLock

import numpy as np
from PIL import Image
from faceswap.core.detector import detect_largest

_ARC2FACE_MODELS = Path("models")
_arc2face_app = None
_arc2face_lock = RLock()


# ---------------------------------------------------------------------------
# Optional Arc2Face ArcFace backbone (antelopev2 / arc2face_arcface.onnx)
# ---------------------------------------------------------------------------

def _try_load_arc2face_app():
    """
    Lazy-load the Arc2Face ArcFace backbone for embedding extraction.
    Returns ("insightface", app) | ("unavailable", None).
    Tries models/arc2face_arcface.onnx then InsightFace antelopev2 (CPU-only).
    """
    global _arc2face_app
    with _arc2face_lock:
        if _arc2face_app is not None:
            return _arc2face_app
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="antelopev2", allowed_modules=["recognition"])
            app.prepare(ctx_id=-1)
            _arc2face_app = ("insightface", app)
        except Exception:
            _arc2face_app = ("unavailable", None)
        return _arc2face_app


def _extract_arc2face_embedding(frame_bgr: np.ndarray) -> np.ndarray | None:
    """Extract antelopev2 embedding for Arc2Face conditioning. None if unavailable."""
    kind, model = _try_load_arc2face_app()
    if kind != "insightface" or model is None:
        return None
    with _arc2face_lock:
        faces = model.get(frame_bgr)
    if not faces:
        return None
    best = max(faces,
               key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = getattr(best, "embedding", None)
    return emb.astype(np.float32) if emb is not None else None


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_embedding_from_image(image_path: str | Path) -> np.ndarray | None:
    """Return raw 512-d buffalo_l embedding of the largest face, or None."""
    with Image.open(image_path) as image:
        frame = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    face = detect_largest(frame)
    return None if face is None else face.embedding.astype(np.float32)


def _extract_dual_embeddings(
    image_path: str | Path,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract both buffalo_l and arc2face embeddings from one image."""
    with Image.open(image_path) as image:
        frame = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    face = detect_largest(frame)
    if face is None:
        return None, None
    buffalo_emb = face.embedding.astype(np.float32)
    arc2face_emb = _extract_arc2face_embedding(frame)
    return buffalo_emb, arc2face_emb


# ---------------------------------------------------------------------------
# Outlier trimming (Fix 1)
# ---------------------------------------------------------------------------

def _trim_outliers(embeddings: list[np.ndarray]) -> list[np.ndarray]:
    """
    Remove embeddings with cosine distance to centroid > median + 1.5 × IQR.
    With ≤ 2 embeddings there is no meaningful IQR, so all are kept.
    """
    if len(embeddings) <= 2:
        return embeddings
    centroid = np.stack(embeddings).mean(axis=0)
    c_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
    dists = np.array([
        1.0 - float(np.dot(e / (np.linalg.norm(e) + 1e-9), c_norm))
        for e in embeddings
    ])
    q1, q3 = np.percentile(dists, 25), np.percentile(dists, 75)
    threshold = float(np.median(dists)) + 1.5 * (q3 - q1)
    survivors = [e for e, d in zip(embeddings, dists) if d <= threshold]
    return survivors if survivors else embeddings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_profile_embedding(image_paths: list[str | Path]) -> np.ndarray:
    """
    Average buffalo_l embeddings with outlier trimming (Fix 1).
    Result magnitude is rescaled to the median original embedding norm so the
    AlphaFace swapper sees a characteristic magnitude — not a shrunken average.
    Returns (512,) float32. Backward-compatible with existing callers and tests.
    """
    embeddings: list[np.ndarray] = []
    for path in image_paths:
        emb = extract_embedding_from_image(path)
        if emb is not None:
            embeddings.append(emb)
        else:
            print(f"[profile] no face detected in {path}")
    if not embeddings:
        raise ValueError("No faces detected in any of the provided images")

    median_norm = float(np.median([np.linalg.norm(e) for e in embeddings]))
    survivors = _trim_outliers(embeddings)
    mean_emb = np.stack(survivors).mean(axis=0).astype(np.float32)

    # Rescale to median original norm (not the collapsed-average norm)
    current_norm = float(np.linalg.norm(mean_emb))
    if current_norm > 1e-9:
        mean_emb = mean_emb * (median_norm / current_norm)
    return mean_emb


def build_profile_embedding_extended(
    image_paths: list[str | Path],
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """
    Dual-space extraction (Fix 2) + quality score (Fix 5).
    Returns (buffalo_embedding, arc2face_embedding | None, quality_score 0-1).
    """
    buffalo_embs: list[np.ndarray] = []
    arc2face_embs: list[np.ndarray] = []
    n_total = len(image_paths)

    for path in image_paths:
        b_emb, a_emb = _extract_dual_embeddings(path)
        if b_emb is not None:
            buffalo_embs.append(b_emb)
            if a_emb is not None:
                arc2face_embs.append(a_emb)
        else:
            print(f"[profile] no face detected in {path}")

    if not buffalo_embs:
        raise ValueError("No faces detected in any of the provided images")

    # --- Quality score (Fix 5): mean pairwise cosine similarity × detection ratio ---
    survivors = _trim_outliers(buffalo_embs)
    n_valid = len(buffalo_embs)
    if len(survivors) >= 2:
        normed = np.stack(survivors)
        normed /= (np.linalg.norm(normed, axis=1, keepdims=True) + 1e-9)
        sim = normed @ normed.T
        idx = np.triu_indices(len(survivors), k=1)
        mean_sim = float(sim[idx].mean()) if idx[0].size > 0 else 1.0
    else:
        mean_sim = 1.0
    detection_ratio = n_valid / max(n_total, 1)
    quality_score = float(np.clip(0.5 * mean_sim + 0.5 * detection_ratio, 0.0, 1.0))

    # --- Buffalo_l embedding (outlier-trimmed, magnitude-rescaled) ---
    median_norm = float(np.median([np.linalg.norm(e) for e in buffalo_embs]))
    mean_emb = np.stack(survivors).mean(axis=0).astype(np.float32)
    current_norm = float(np.linalg.norm(mean_emb))
    if current_norm > 1e-9:
        mean_emb = mean_emb * (median_norm / current_norm)
    buffalo_embedding = mean_emb

    # --- Arc2Face embedding (Fix 2, optional) ---
    arc2face_embedding: np.ndarray | None = None
    if arc2face_embs:
        a_survivors = _trim_outliers(arc2face_embs)
        a_med_norm = float(np.median([np.linalg.norm(e) for e in arc2face_embs]))
        a_mean = np.stack(a_survivors).mean(axis=0).astype(np.float32)
        a_norm = float(np.linalg.norm(a_mean))
        if a_norm > 1e-9:
            a_mean = a_mean * (a_med_norm / a_norm)
        arc2face_embedding = a_mean

    return buffalo_embedding, arc2face_embedding, quality_score


def save_profile(embedding: np.ndarray, save_path: str | Path,
                 *, arc2face_embedding: np.ndarray | None = None,
                 quality_score: float | None = None) -> None:
    kwargs: dict = {"embedding": embedding}
    if arc2face_embedding is not None:
        kwargs["arc2face_embedding"] = arc2face_embedding
    if quality_score is not None:
        kwargs["quality_score"] = np.array(quality_score, dtype=np.float32)
    np.savez(str(save_path), **kwargs)


def load_profile(npz_path: str | Path) -> np.ndarray:
    """Return raw buffalo_l embedding. Preserves magnitude required by AlphaFace."""
    with np.load(str(npz_path)) as data:
        return data["embedding"].astype(np.float32)


def load_profile_extended(
    npz_path: str | Path,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Return (buffalo_embedding, arc2face_embedding | None, quality_score)."""
    with np.load(str(npz_path)) as data:
        emb = data["embedding"].astype(np.float32)
        a_emb = (data["arc2face_embedding"].astype(np.float32)
                 if "arc2face_embedding" in data else None)
        qs = float(data["quality_score"]) if "quality_score" in data else 0.0
    return emb, a_emb, qs
