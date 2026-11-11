# Real-time quality repair plan

Goal: implement the user's 20 requested repairs while retaining a 33 ms target.
Architecture: 256px ArcFace swap, homogeneous reprojection into FFHQ512, budgeted restoration, masked color/detail processing, Laplacian paste, face-only motion compensation. Latest-only queues bound latency. Level 2 diffusion runs once at source selection.

1. Image contracts and alignment (aligner, swapper, color, blend): regression tests for scale, raw identity, input/output ranges, and exterior preservation.
2. Pipeline and temporal: regression tests for newest-item queues, transform composition, face-only EMA, reset, and restoration budget recovery.
3. Profiles and runtime dependencies: shared buffalo_l, correct Arc2Face conditioning and LoRA loading; validate mocked contracts and syntax.
4. GUI/camera: signal-only UI mutation, atomic initialization, capture mailbox and cleanup, dtype safety; test fake devices.
5. Integration: run unittest discovery and requested AST command; document absent model/hardware validation.

Choices: blink detection disabled with five-point landmarks; no inter-eye-distance heuristic. Budget uses rolling measurements plus last frame; periodically re-evaluates restoration cost without reusing a stale enhanced face. Phone output remains an explicit stub.
