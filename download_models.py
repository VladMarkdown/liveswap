"""
Download all required liveswap models to models/.
Run once before launching the application:
    python download_models.py

NOTE: URLs marked PLACEHOLDER below need to be replaced with real model URLs
when the models are publicly hosted. The download infrastructure is complete.
"""
from __future__ import annotations
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# Try to import tqdm, install if missing
try:
    from tqdm import tqdm
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
    from tqdm import tqdm  # type: ignore

MODELS_DIR = Path("models")

# (filename, url, description)
# PLACEHOLDER URLs: replace with real hosted URLs before distributing
DIRECT_MODELS: list[tuple[str, str, str]] = [
    (
        "alphaface_256.onnx",
        # PLACEHOLDER — replace with actual AlphaFace_256 download URL
        "https://huggingface.co/deepinsight/alphaface/resolve/main/alphaface_256.onnx",
        "AlphaFace_256 face swap model (~190 MB)",
    ),
    (
        "GPEN-BFR-512.onnx",
        # PLACEHOLDER — replace with actual GPEN-BFR-512 download URL
        "https://huggingface.co/bes-dev/GPEN/resolve/main/GPEN-BFR-512.onnx",
        "GPEN-BFR-512 face enhancer (~240 MB)",
    ),
    (
        "bisenet.onnx",
        # PLACEHOLDER — replace with actual BiSeNet face parser URL
        "https://huggingface.co/deepinsight/bisenet/resolve/main/bisenet.onnx",
        "BiSeNet face segmentation (~50 MB)",
    ),
    (
        "arc2face_arcface.onnx",
        # PLACEHOLDER — Arc2Face ArcFace backbone (antelopev2 R100 / WebFace42M)
        "https://huggingface.co/FoivosPar/Arc2Face/resolve/main/arcface_model/arcface_w600k_r50.onnx",
        "Arc2Face ArcFace backbone (~165 MB)",
    ),
    (
        "softcam.dll",
        # PLACEHOLDER — tshino/softcam prebuilt DLL; install separately if not available
        "https://github.com/tshino/softcam/releases/latest/download/softcam.dll",
        "Softcam virtual DirectShow camera (~1 MB)",
    ),
]

INSIGHTFACE_PACKS: list[tuple[str, str]] = [
    ("buffalo_l",   "InsightFace buffalo_l: detector + ArcFace + 106-pt (~300 MB)"),
    ("antelopev2",  "InsightFace antelopev2: ArcFace R100 for Arc2Face (~300 MB)"),
]

APPROX_SIZES_MB: dict[str, int] = {
    "alphaface_256.onnx":  190,
    "GPEN-BFR-512.onnx":   240,
    "bisenet.onnx":          50,
    "arc2face_arcface.onnx": 165,
    "softcam.dll":             1,
    "buffalo_l":             300,
    "antelopev2":            300,
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "liveswap-downloader/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0)) or None
            with open(tmp, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                desc=dest.name[:40], leave=False,
            ) as bar:
                while chunk := resp.read(1 << 16):
                    f.write(chunk)
                    bar.update(len(chunk))
        shutil.move(str(tmp), str(dest))
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _download_insightface_pack(name: str) -> bool:
    """Let InsightFace download and cache the pack automatically."""
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name=name, allowed_modules=["recognition"])
        app.prepare(ctx_id=-1)
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


def _size_str(mb: int) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb} MB"


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)

    already_mb = sum(
        APPROX_SIZES_MB.get(name, 0)
        for name in APPROX_SIZES_MB
        if (MODELS_DIR / name).exists()
    )
    total_mb = sum(APPROX_SIZES_MB.values())
    need_mb = total_mb - already_mb

    print()
    print("liveswap model downloader")
    print(f"  Total: ~{_size_str(total_mb)}")
    print(f"  Already downloaded: ~{_size_str(already_mb)}")
    print(f"  To download: ~{_size_str(need_mb)}")
    print()

    results: dict[str, bool] = {}

    # --- Direct ONNX model downloads ---
    for fname, url, desc in DIRECT_MODELS:
        dest = MODELS_DIR / fname
        if dest.exists():
            print(f"  ✓ {fname}  (already present)")
            results[fname] = True
            continue
        print(f"  ↓ {fname}  {desc}")
        try:
            _download(url, dest)
            print(f"  ✓ {fname} — ready")
            results[fname] = True
        except Exception as e:
            print(f"  ✗ {fname} — FAILED: {e}")
            if "PLACEHOLDER" in url or "huggingface.co" in url:
                print(f"    This URL is a placeholder. Download manually:")
                print(f"    Place the file at: {dest.resolve()}")
            results[fname] = False

    print()

    # --- InsightFace model packs ---
    for pack_name, desc in INSIGHTFACE_PACKS:
        pack_dir = MODELS_DIR / pack_name
        if pack_dir.exists() and any(pack_dir.iterdir()):
            print(f"  ✓ {pack_name}  (already present)")
            results[pack_name] = True
            continue
        print(f"  ↓ {pack_name}  {desc}")
        ok = _download_insightface_pack(pack_name)
        if ok:
            print(f"  ✓ {pack_name} — ready")
        else:
            print(f"  ✗ {pack_name} — FAILED (see error above)")
        results[pack_name] = ok

    # --- Summary ---
    print()
    n_ok   = sum(1 for v in results.values() if v)
    n_fail = len(results) - n_ok
    print("=" * 50)
    print(f"Ready: {n_ok}/{len(results)} models")
    if n_fail:
        print(f"Failed: {n_fail}")
        print("Replace PLACEHOLDER URLs in download_models.py with real model URLs,")
        print("or download the files manually and place them in models/.")
        return 1
    print("All models ready. Run:  python main.py")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
