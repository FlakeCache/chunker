# Chunker - Claude Code Instructions

High-performance content-defined chunking library for Nix NARs.

## Build Commands

Use `just` for all commands - auto-enters nix develop:

```bash
just build        # cargo build
just build-release # cargo build --release
just test         # cargo test --all
just test-otel    # cargo test with otel feature
just test-nif     # cargo test with nif feature
just clippy       # run clippy on all feature combos
just fmt          # cargo fmt
just fmt-check    # cargo fmt --check
just ci           # full CI check locally
just --list       # show all commands
```

## Features

- `sha2-asm` - Assembly-optimized SHA2 (default)
- `nif` - Erlang/Elixir NIF bindings
- `otel` - OpenTelemetry + Tokio runtime
- `async-stream` - Async streaming support

Note: `nif` and `otel` are mutually exclusive.

## Git Hooks

Lefthook runs automatically on commit/push:
- pre-commit: fmt + clippy (parallel)
- pre-push: tests (parallel)

## First-time Setup

Run once per machine:
```bash
./scripts/setup-dev.sh
```
