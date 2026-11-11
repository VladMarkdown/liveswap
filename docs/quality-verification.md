# Quality stack verification

Implemented the requested model contracts, 256px alignment, homogeneous FFHQ
reprojection, Laplacian paste, gamma-correct detail, BGR scattering, face-only
backward-flow temporal history, latest-only queues, GPEN budget statistics,
shared raw buffalo_l profiles, Arc2Face-conditioned LoRA training and one-time
source generation, GUI signal/lifecycle fixes, and native camera dtype guards.

CaptureThread is used by the desktop GUI. IPCameraCapture offers the same read
interface for Phase 2. WebRTCOutput intentionally raises NotImplementedError;
no phone streaming implementation is claimed. Blink detection is explicitly
disabled because five landmarks do not measure eyelid closure. Eye/mouth pixels
are excluded from temporal history blending.

Validation on this host:
- Requested AST command: passed (syntax check done).
- compileall faceswap/tests: passed.
- unittest discovery: 26 tests; 12 passed, 14 explicitly skipped.
- Missing numpy/OpenCV/Pillow prevent numerical image/profile tests.
- Dependency installation was attempted but network access was blocked.
- models/ contains no inference checkpoints; webcam/virtual camera, GPU quality,
  Arc2Face training, and the 30 fps target remain unverified.

Run after installing the selected runtime and model dependencies:

    python -m unittest discover -s tests -v
    python main.py

For RTX 3070, select the commented CUDA runtime in requirements.txt and remove
the CPU runtime line. Do not install several ONNX Runtime distributions together.
Measure pipeline.last_inference_ms, pipeline.average_inference_ms and
gpen_skipped with the real models at 1280x720. Test face entry/exit, head motion,
blinks, glasses, mouth motion, camera disconnect and repeated start/stop.
No measured quality score or 30 fps guarantee is asserted.

See profile_implementation.md for the buffalo_l/Arc2Face embedding compatibility
risk and the unverified arc2face>=0.1.0 package availability. The requested
requirement is present, but upstream installation may need its repository module.
