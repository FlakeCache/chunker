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

# Run tests with telemetry feature
test-telemetry:
    cargo test --all --features telemetry

# Run tests with nif feature
test-nif:
    cargo test --all --features nif

# Run clippy on all feature combinations
clippy:
    cargo clippy -- -D warnings
    cargo clippy --features telemetry -- -D warnings
    cargo clippy --features nif -- -D warnings

# Format code
fmt:
    cargo fmt

# Check formatting
fmt-check:
    cargo fmt -- --check

# Run benchmarks
bench:
    RUSTFLAGS="-C target-cpu=native" cargo bench
    BENCH_RUSTFLAGS="-C target-cpu=native" scripts/export-criterion.py

# Run a shorter benchmark pass and export summaries
bench-quick:
    cargo bench --bench throughput -- --sample-size 10
    scripts/export-criterion.py

# Run a shorter native-CPU benchmark pass and export summaries
bench-quick-native:
    RUSTFLAGS="-C target-cpu=native" cargo bench --bench throughput -- --sample-size 10
    BENCH_RUSTFLAGS="-C target-cpu=native" scripts/export-criterion.py

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
ci: fmt-check clippy test test-telemetry test-nif
    @echo "All CI checks passed!"
