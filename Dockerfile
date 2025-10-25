# Multi-stage Dockerfile for LTEngine (CPU)
FROM rust:1.72-slim AS builder

# Install system deps required by native crates (adjust as needed)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake pkg-config libssl-dev libopenblas-dev ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/ltengine

# Copy Cargo manifests first to leverage Docker cache
COPY Cargo.toml Cargo.lock ./
# Copy the crate sources
COPY ltengine ./ltengine

# Build release (uses the workspace/crate binary name)
RUN cargo build --release --manifest-path ltengine/Cargo.toml

# ---- runtime image ----
FROM debian:bullseye-slim

# Minimal runtime deps (add libs your binary needs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates libopenblas3 && \
    rm -rf /var/lib/apt/lists/*

# Copy the compiled binary from builder
COPY --from=builder /usr/src/ltengine/ltengine/target/release/ltengine /usr/local/bin/ltengine

ENV MODEL_PATH=/models
VOLUME ["/models"]

EXPOSE 8080

CMD ["/usr/local/bin/ltengine"]