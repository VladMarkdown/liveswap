# Profile implementation notes

Profile extraction and live inference share one locked `buffalo_l` detector. New
profiles store the mean of raw embeddings. Existing normalized NPZ profiles have
already lost their magnitude; recreate them from the original photos when raw
AlphaFace embeddings are needed.

Arc2Face input is normalized at the diffusion boundary. Its upstream instructions
require a custom WebFace42M `arcface.onnx`, replacing the default antelopev2
recognition backbone. The requested buffalo_l sharing does not reproduce that
embedding distribution. Identity similarity with Arc2Face therefore needs model
validation; matching vector length alone does not establish compatibility.
[Upstream model and inference instructions](https://github.com/foivospar/Arc2Face#download-models).

The trainer uses Arc2Face's `arc2face` UNet and projected identity conditioning.
It freezes the base weights, trains attention LoRA adapters, and saves diffusers
LoRA weights. Session setup loads those weights, synthesizes a portrait and
extracts a source Face once, then releases diffusion memory. GPU generation and
training have not been exercised in the dependency-free development environment.
Synthetic-only training does not establish any particular quality improvement.

`arc2face>=0.1.0` is retained in requirements as requested. A matching PyPI
release could not be verified. Upstream documents a repository checkout with its
local `arc2face` module, rather than a versioned PyPI installation. If the package
cannot resolve, use the upstream module on Python's import path and install its
requirements; do not assume an unrelated package named arcface is compatible.
[Upstream installation](https://github.com/foivospar/Arc2Face#installation).

Install only one ONNX Runtime variant. CPU is the default; uncomment CUDA for NVIDIA or DirectML for AMD/Intel,
removing the CPU runtime line when selecting either GPU alternative. Torch diffusion uses CUDA when available and otherwise
CPU; choosing ONNX DirectML does not enable DirectML for diffusion. xformers is a
CUDA dependency and can be omitted for CPU/DirectML installations.
