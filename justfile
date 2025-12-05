# Justfile - all commands auto-enter nix develop
# Install just: nix profile install nixpkgs#just
# Usage: just build, just test, just clippy

set shell := ["nix", "develop", "-c", "bash", "-c"]

# Default: show available commands
default:
    @just --list

# Build the library
build:
    cargo build

# Build release
build-release:
    cargo build --release

# Run all tests
test:
    cargo test --all

# Run tests with otel feature
test-otel:
    cargo test --all --features otel

# Run tests with nif feature
test-nif:
    cargo test --all --features nif

# Run clippy on all feature combinations
clippy:
    cargo clippy -- -D warnings
    cargo clippy --features otel -- -D warnings
    cargo clippy --features nif -- -D warnings

# Format code
fmt:
    cargo fmt

# Check formatting
fmt-check:
    cargo fmt -- --check

# Run benchmarks
bench:
    cargo bench

# Clean build artifacts
clean:
    cargo clean

# Run fuzz tests (requires cargo-fuzz)
fuzz:
    cd fuzz && cargo fuzz run chunking --max-len=1000000 -- -max_total_time=10
    cd fuzz && cargo fuzz run compression --max-len=10000000 -- -max_total_time=10
    cd fuzz && cargo fuzz run decompression --max-len=10000000 -- -max_total_time=10
    cd fuzz && cargo fuzz run signing --max-len=1000 -- -max_total_time=10

# Full CI check (what CI runs)
ci: fmt-check clippy test test-otel test-nif
    @echo "All CI checks passed!"
