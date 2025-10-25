```markdown
# Running LTEngine with Docker

This document explains how to build and run LTEngine in Docker.

Prerequisites
- Docker engine (and docker-compose if you want to use the compose example)
- If you plan to use GPU/CUDA builds: NVIDIA drivers + nvidia-container-toolkit

CPU (build & run)
1. Build the image:
   docker build -t libretranslate/ltengine:local .

2. Prepare a models directory:
   mkdir -p ./models
   # Place your model files under ./models and set MODEL_PATH to point there

3. Run:
   docker run --rm -p 8080:8080 -v $(pwd)/models:/models -e MODEL_PATH=/models libretranslate/ltengine:local

Using docker-compose
1. Edit docker-compose.yml to point to your models folder if needed.
2. Run:
   docker compose up --build

GPU (CUDA)
- If GPU support exists in LTEngine, use docker/Dockerfile.cuda and run with:
  docker run --gpus all -v $(pwd)/models:/models -e MODEL_PATH=/models libretranslate/ltengine:gpu

Notes
- Do NOT bake model files into the image. Mount host volumes or use network storage.
- If the crate binary name differs from "ltengine", update the Dockerfile COPY and CMD accordingly.
```