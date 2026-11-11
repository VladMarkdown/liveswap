"""
Download all required liveswap models to models/.
Run once before launching the application:
    python download_models.py

Live mode needs the swap, restoration and parsing models plus the buffalo_l
pack. The Arc2Face files are only used by profile mode; a failure there is
reported but does not fail the run.

Console output is ASCII only: a redirected stdout on Windows uses the legacy
code page and cannot encode symbols such as check marks.
"""
from __future__ import annotations
import ctypes
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# Try to import tqdm, install if missing
try:
    from tqdm import tqdm
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
    from tqdm import tqdm  # type: ignore

MODELS_DIR = Path("models")

# (filename, url, description, required for live mode)
DIRECT_MODELS: list[tuple[str, str, str, bool]] = [
    (
        "hyperswap_1b_256.onnx",
        "https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1b_256.onnx",
        "HyperSwap 1b 256 face swap model",
        True,
    ),
    (
        "GPEN-BFR-512.onnx",
        "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gpen_bfr_512.onnx",
        "GPEN-BFR-512 face enhancer (~240 MB)",
        True,
    ),
    (
        "bisenet.onnx",
        "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/bisenet_resnet_34.onnx",
        "BiSeNet face segmentation (~50 MB)",
        True,
    ),
    (
        "arc2face_arcface.onnx",
        "https://huggingface.co/FoivosPar/Arc2Face/resolve/main/arcface.onnx",
        "Arc2Face ArcFace backbone, profile mode only",
        False,
    ),
]
# softcam is NOT in DIRECT_MODELS - it needs COM registration, handled by install_softcam()

# (pack name, description, required for live mode)
INSIGHTFACE_PACKS: list[tuple[str, str, bool]] = [
    ("buffalo_l",   "InsightFace buffalo_l: detector + ArcFace + 106-pt (~300 MB)", True),
    ("antelopev2",  "InsightFace antelopev2: ArcFace R100, profile mode only (~300 MB)", False),
]

APPROX_SIZES_MB: dict[str, int] = {
    "hyperswap_1b_256.onnx": 400,
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


_SOFTCAM_API = "https://api.github.com/repos/tshino/softcam/releases/latest"
_SOFTCAM_DLL  = MODELS_DIR / "softcam.dll"


def _softcam_is_available() -> bool:
    """Return True if softcam is already usable (DLL present or already registered)."""
    if _SOFTCAM_DLL.exists():
        return True
    if Path(r"C:\Windows\System32\softcam.dll").exists():
        return True
    try:
        ctypes.cdll.LoadLibrary("softcam")
        return True
    except OSError:
        return False


def install_softcam() -> bool:
    """
    Download and register the softcam virtual DirectShow camera.
    Returns True if softcam is available after the call.
    Never raises - softcam is optional (app works without it, just no virtual camera).
    """
    if _softcam_is_available():
        print("  + softcam -- already installed")
        return True

    print("  v softcam -- downloading virtual camera driver (tshino/softcam)...")

    # --- Fetch latest release metadata from GitHub API ---
    try:
        req = urllib.request.Request(
            _SOFTCAM_API,
            headers={"User-Agent": "liveswap-downloader/1.0",
                     "Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            release = json.loads(resp.read())
    except Exception as e:
        print("  ! softcam not installed.")
        print("    For virtual camera, run download_models.py with an internet connection.")
        print(f"    (network error: {e})")
        return False

    assets = release.get("assets", [])
    installer = next((a for a in assets if a["name"].endswith(".exe")), None)
    zip_asset  = next((a for a in assets if a["name"].endswith(".zip")), None)

    if installer is None and zip_asset is None:
        print("  ! softcam release has no installable assets.")
        print("    Install manually from https://github.com/tshino/softcam/releases")
        return False

    MODELS_DIR.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        try:
            if installer:
                # NSIS installer - silent flag /S
                inst_path = tmp / installer["name"]
                _download(installer["browser_download_url"], inst_path)
                subprocess.run([str(inst_path), "/S"], check=True)
                # Installer registers the DLL; it may land in System32
            else:
                # ZIP archive - extract DLL, copy to models/, register with regsvr32
                zip_path = tmp / zip_asset["name"]
                _download(zip_asset["browser_download_url"], zip_path)
                with zipfile.ZipFile(zip_path) as zf:
                    dll_entries = [n for n in zf.namelist()
                                   if n.lower().endswith("softcam.dll")]
                    if not dll_entries:
                        print("  ! softcam.dll not found inside the downloaded zip.")
                        return False
                    zf.extract(dll_entries[0], tmp)
                    extracted = tmp / dll_entries[0]
                shutil.copy2(str(extracted), str(_SOFTCAM_DLL))
                result = subprocess.run(
                    ["regsvr32", "/s", str(_SOFTCAM_DLL.resolve())],
                    capture_output=True,
                )
                if result.returncode != 0:
                    print("  ! softcam needs administrator rights to register.")
                    print("    Right-click download_models.py -> Run as administrator")
                    # DLL is copied to models/ so virtual_cam.py can still try to load it
                    return False
        except subprocess.CalledProcessError:
            print("  ! softcam needs administrator rights to install.")
            print("    Right-click download_models.py -> Run as administrator")
            return False
        except Exception as e:
            print(f"  ! softcam install failed: {e}")
            return False

    # Verify the DLL is loadable
    try:
        ctypes.cdll.LoadLibrary(str(_SOFTCAM_DLL.resolve()))
    except OSError:
        pass  # May work after a restart even if LoadLibrary fails here

    print("  + softcam installed -- DirectShow Softcam camera is now available")
    return True


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
    required: set[str] = set()

    # --- Direct ONNX model downloads ---
    for fname, url, desc, needed in DIRECT_MODELS:
        if needed:
            required.add(fname)
        dest = MODELS_DIR / fname
        if dest.exists():
            print(f"  + {fname}  (already present)")
            results[fname] = True
            continue
        print(f"  v {fname}  {desc}")
        try:
            _download(url, dest)
            print(f"  + {fname} - ready")
            results[fname] = True
        except Exception as e:
            print(f"  ! {fname} - FAILED: {e}")
            print(f"    Download it manually and place it at: {dest.resolve()}")
            results[fname] = False

    print()

    # --- InsightFace model packs ---
    for pack_name, desc, needed in INSIGHTFACE_PACKS:
        if needed:
            required.add(pack_name)
        # InsightFace keeps its packs under ~/.insightface/models.
        pack_dir = Path.home() / ".insightface" / "models" / pack_name
        if pack_dir.exists() and any(pack_dir.iterdir()):
            print(f"  + {pack_name}  (already present)")
            results[pack_name] = True
            continue
        print(f"  v {pack_name}  {desc}")
        ok = _download_insightface_pack(pack_name)
        if ok:
            print(f"  + {pack_name} - ready")
        else:
            print(f"  ! {pack_name} - FAILED (see error above)")
        results[pack_name] = ok

    # --- Softcam virtual camera ---
    print()
    softcam_ok = install_softcam()

    # --- Summary ---
    print()
    failed = [name for name, ok in results.items() if not ok]
    blocking = [name for name in failed if name in required]
    print("=" * 50)
    print(f"Models ready: {len(results) - len(failed)}/{len(results)}")
    if softcam_ok:
        print("Virtual camera: installed")
    else:
        print("Virtual camera: not installed (the app still runs and shows its preview)")
    for name in failed:
        kind = "REQUIRED" if name in required else "optional, profile mode only"
        print(f"Failed: {name} ({kind})")
    if blocking:
        print("Download the required files manually and place them in models/.")
        return 1
    print("Run:  python main.py")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
