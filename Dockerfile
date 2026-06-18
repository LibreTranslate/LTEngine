# syntax=docker/dockerfile:1

# Build (CUDA dev toolkit + Rust + clang/cmake for llama.cpp)
FROM nvidia/cuda:13.3.0-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential clang cmake pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/local/cuda/include/nccl*.h

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

# Copy LTEngine source
COPY . .

# Build with CUDA support (cargo caches survive layer cache hits)
RUN --mount=type=cache,id=ltengine-cargo-git,target=/usr/local/cargo/git \
    --mount=type=cache,id=ltengine-cargo-registry,target=/root/.cargo/registry \
    --mount=type=cache,id=ltengine-target-cuda13-nonccl,target=/build/target \
    cargo build --features cuda --release -p ltengine && \
    cp target/release/ltengine /ltengine

# Runtime (lean: only CUDA runtime libs, no build tools)
FROM nvidia/cuda:13.3.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --no-create-home ltengine \
    && mkdir -p /models \
    && chown ltengine:ltengine /models

COPY --from=builder /ltengine /usr/local/bin/ltengine

ENV HF_HOME=/models
ENV LTENGINE_MODEL=gemma3-4b
VOLUME ["/models"]
EXPOSE 5050

USER ltengine
CMD ["sh", "-c", "exec ltengine --host 0.0.0.0 -m \"${LTENGINE_MODEL}\" ${LTENGINE_MODEL_FILE:+--model-file \"${LTENGINE_MODEL_FILE}\"}"]
