"""Post-install check: load each model once and report where it runs.

Prints one line per model and a final verdict. Exits 0 even on a CPU fallback
(the app still works, slowly) and non-zero only if a model cannot load at all.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                        format="%(name)s %(message)s")
    import onnxruntime as ort
    from faceswap.core.ort_utils import create_session

    print(f"Python {sys.version.split()[0]}, onnxruntime {ort.__version__}")
    models = Path("models")
    names = ["GPEN-BFR-512.onnx", "bisenet.onnx"]
    names.append("alphaface_256.onnx" if (models / "alphaface_256.onnx").exists()
                 else "hyperswap_1b_256.onnx")
    on_gpu = True
    for name in names:
        provider = create_session(models / name).get_providers()[0]
        on_gpu = on_gpu and provider != "CPUExecutionProvider"
        print(f"  {name}: {provider}")

    import insightface
    print(f"insightface {insightface.__version__}")

    if on_gpu:
        print("RESULT: OK, the models run on the graphics card.")
    else:
        print("RESULT: WARNING, the models run on the CPU. Update the NVIDIA "
              "driver and run the installer again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
