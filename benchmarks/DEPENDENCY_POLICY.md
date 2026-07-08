# Dependency Policy

Purpose: keep hashing and chunking upgrades evidence-driven instead of accepting
semver-major changes that weaken cache compatibility or hot-path performance.

## Hashing

- SHA-256 is the default hash for `chunk_data`, `chunk_descriptors`,
  `ChunkStream::new`, and NIF chunking APIs.
- Direct SHA-256 hashing uses `sha2 = "0.11.0"`. `sha2 0.11` has no `asm`
  feature; do not reintroduce the removed `sha2-asm` feature without proving the
  selected backend exists and improves the cache hot path.
- Any SHA backend change must preserve default SHA-256 semantics and include a
  native benchmark comparison against `benchmarks/latest.md`.
- BLAKE3 is supported as an explicit hash algorithm for internal or protocol
  surfaces that allow non-SHA chunk hashes. Do not make it the default without a
  storage/protocol compatibility decision.

## Chunk Boundaries

- Direct chunking uses `fastcdc = "4.0.1"`.
- The v4 upgrade accepted a measured raw chunker regression because keeping the
  dependency current was chosen over holding v3. Keep that tradeoff visible in
  `benchmarks/latest.md`.
- FastCDC major upgrades must run a boundary fixture comparison and a native
  benchmark comparison before acceptance. Do not accept a boundary-changing
  upgrade without a storage compatibility decision.

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
