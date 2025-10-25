
# Running LTEngine with Docker (CPU, Vulkan, CUDA)

LTEngine supports GGUF models compatible with llama.cpp and can be built with backend features:
- CPU (default)
- Vulkan: `--features vulkan`
- CUDA: `--features cuda` (requires CUDA toolkit to build and NVIDIA drivers/Toolkit at runtime)
- Metal: macOS-only; see notes below.

Models
- Pass a path to a custom GGUF file via the CLI `--model-file /models/custom.gguf`.
- If not provided, LTEngine may download a preset model when using `-m <model>`.

Build & run examples

1) CPU (no special features)
   docker build -t denyncrawford/ltengine:cpu .
   mkdir -p models
   docker run --rm -p 8080:8080 -v $(pwd)/models:/models denyncrawford/ltengine:cpu --model-file /models/custom.gguf

2) Vulkan
   # Add libvulkan-dev is installed in the builder when FEATURES=vulkan
   docker build --build-arg FEATURES=vulkan -t denyncrawford/ltengine:vulkan .
   docker run --rm -p 8080:8080 -v $(pwd)/models:/models -v /dev/dri:/dev/dri denyncrawford/ltengine:vulkan --model-file /models/custom.gguf

   Note: give access to GPU devices with `-v /dev/dri:/dev/dri` and ensure host Vulkan drivers are installed.

3) CUDA
   docker build \
     --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 \
     --build-arg RUNTIME_BASE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
     --build-arg FEATURES=cuda \
     -t denyncrawford/ltengine:cuda .
   docker run --gpus all --rm -p 8080:8080 -v $(pwd)/models:/models denyncrawford/ltengine:cuda --model-file /models/custom.gguf

   Requirements:
    - Build uses a CUDA *devel* image that includes nvcc/headers.
    - Runtime requires proper NVIDIA drivers and nvidia-container-toolkit on the host.

4) Metal (macOS)
   - Metal is macOS-only. Build on macOS:
     cargo build --release --manifest-path ltengine/Cargo.toml --features metal
   - To distribute a macOS binary, build on macOS (or a macOS runner in CI) and publish an artifact.

Testing endpoints
- Languages:
  curl http://localhost:8080/languages
- Translate:
  curl -X POST "http://localhost:8080/translate" -H "Content-Type: application/json" -d '{"q":"Hello","source":"en","target":"es","format":"text"}'

Notes
- Do NOT bake large models into the image — mount them at /models.
- For GPU builds ensure build/runtime CUDA or Vulkan versions match host support.