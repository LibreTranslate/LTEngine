# syntax=docker/dockerfile:1

# Keep in sync with the llama-cpp-rs submodule pin in .gitmodules
ARG LLAMA_CPP_RS_COMMIT=20f6e2e652b6e4f859468737a54ecc02597e37f0

# Fetch llama-cpp-rs submodule (pinned commit, separate cached layer)
FROM alpine/git AS submodule
ARG LLAMA_CPP_RS_COMMIT
RUN git clone --filter=blob:none https://github.com/pierotofy/llama-cpp-rs /llama-cpp-rs \
    && git -C /llama-cpp-rs checkout ${LLAMA_CPP_RS_COMMIT} \
    && git -C /llama-cpp-rs submodule update --init --recursive

# Build (CUDA dev toolkit + Rust + clang/cmake for llama.cpp)
FROM nvidia/cuda:12.9.2-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential clang cmake pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

# Copy pinned submodule (cached; only re-runs when LLAMA_CPP_RS_COMMIT changes)
COPY --from=submodule /llama-cpp-rs /build/llama-cpp-rs

# Copy LTEngine source
COPY . .

# Build with CUDA support (cargo caches survive layer cache hits)
RUN --mount=type=cache,id=ltengine-cargo-git,target=/usr/local/cargo/git \
    --mount=type=cache,id=ltengine-cargo-registry,target=/usr/local/cargo/registry \
    --mount=type=cache,id=ltengine-target,target=/build/target \
    cargo build --features cuda --release -p ltengine && \
    cp target/release/ltengine /ltengine

# Runtime (lean: only CUDA runtime libs, no build tools)
FROM nvidia/cuda:12.9.2-runtime-ubuntu24.04

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
CMD ["sh", "-c", "exec ltengine --host 0.0.0.0 -m \"${LTENGINE_MODEL}\""]
