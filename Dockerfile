# Multi-stage build supporting configurable features: cuda, vulkan
# Build usage examples:
#  - CPU only: docker build -t ltengine:cpu .
#  - Vulkan:   docker build --build-arg FEATURES=vulkan -t ltengine:vulkan .
#  - CUDA:     docker build \
#       --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 \
#       --build-arg RUNTIME_BASE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
#       --build-arg FEATURES=cuda \
#       -t ltengine:cuda .
ARG BASE_IMAGE=rust:1.72-slim
FROM ${BASE_IMAGE} AS builder

ARG FEATURES=""
ARG RUSTFLAGS=""
ENV RUSTFLAGS=${RUSTFLAGS}
ENV DEBIAN_FRONTEND=noninteractive

# Install common build deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential cmake pkg-config libssl-dev libopenblas-dev ca-certificates git curl \
      libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libx11-dev libxrandr-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/ltengine

# Copy manifests to leverage caching
COPY Cargo.toml Cargo.lock ./

# Copy source
COPY ltengine ./ltengine

# Conditionally install Vulkan SDK headers if building with vulkan feature
RUN if echo "$FEATURES" | grep -qw vulkan; then \
      apt-get update && apt-get install -y --no-install-recommends libvulkan-dev vulkan-utils && \
      rm -rf /var/lib/apt/lists/*; \
    fi

# Note: For CUDA builds, use a CUDA *devel* base image via --build-arg BASE_IMAGE=nvidia/cuda:...-devel
# That image already contains nvcc and CUDA headers; if additional packages are required add them here.

# Build with features (if FEATURES is empty, cargo will ignore --features)
RUN if [ -z "$FEATURES" ]; then \
      cargo build --release --manifest-path ltengine/Cargo.toml ; \
    else \
      cargo build --release --manifest-path ltengine/Cargo.toml --features "$FEATURES" ; \
    fi

# ---- runtime image ----
ARG RUNTIME_BASE=debian:bullseye-slim
FROM ${RUNTIME_BASE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
# Small runtime deps; add runtime libs required by features (e.g., libvulkan1 or CUDA runtime)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates libopenblas3 libxcb1 && \
    rm -rf /var/lib/apt/lists/*

# Copy binary (adjust if binary name differs)
COPY --from=builder /usr/src/ltengine/ltengine/target/release/ltengine /usr/local/bin/ltengine
RUN chmod +x /usr/local/bin/ltengine

ENV MODEL_PATH=/models
VOLUME ["/models"]
EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/ltengine"]