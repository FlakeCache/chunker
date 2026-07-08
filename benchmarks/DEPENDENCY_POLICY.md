# Dependency Policy

Purpose: keep hashing and chunking upgrades evidence-driven instead of accepting
semver-major changes that weaken cache compatibility or hot-path performance.

## Hashing

- SHA-256 is the default hash for `chunk_data`, `chunk_descriptors`,
  `ChunkStream::new`, and NIF chunking APIs.
- Keep direct SHA-256 hashing on `sha2 = "0.10.9"` while this crate depends on
  `sha2/asm` through the default `sha2-asm` feature.
- Do not migrate the direct SHA-256 path to `sha2 0.11` unless the replacement
  provides an equivalent accelerated backend or benchmarks prove the loss is
  acceptable for the cache hot path.
- `sha2 0.11` is acceptable transitively for dependencies such as
  `ed25519-dalek`; it must not become the direct chunk hashing backend by
  accident.
- BLAKE3 is supported as an explicit hash algorithm for internal or protocol
  surfaces that allow non-SHA chunk hashes. Do not make it the default without a
  storage/protocol compatibility decision.

## Chunk Boundaries

- Keep `fastcdc = "3.2.1"` until a newer release matches both boundary behavior
  and same-host throughput.
- Any FastCDC major upgrade must run a boundary fixture comparison and a native
  benchmark comparison before acceptance.

## Required Checks For Hashing Or Chunking Upgrades

Run:

```bash
cargo bench --bench throughput --no-run
cargo test --all
cargo test --all --features nif
cargo test --all --features async-stream
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features nif -- -D warnings
cargo fmt --all -- --check
git diff --check
```

For performance-sensitive upgrades, also run:

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench --bench throughput -- --sample-size 10
BENCH_RUSTFLAGS="-C target-cpu=native" scripts/export-criterion.py
```

