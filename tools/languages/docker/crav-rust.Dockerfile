# Rust analysis image: the official toolchain (check/test/build) + clippy. The
# `rust` image usually ships clippy already; the component add is a no-op if so.
# The cargo registry cache is mounted at run time (/usr/local/cargo/registry).
FROM rust:1-slim
RUN rustup component add clippy
