# Chunker

High-performance content-defined chunking (FastCDC) for Nix NARs.

## What it does
- FastCDC chunking with SHA-256 (default) or BLAKE3 hashing
- Compression: zstd, xz (LZMA2), bzip2
- Ed25519 signing and Nix base32 encoding
- Optional Rustler NIF bindings for Elixir

## Quick start (Rust)
```rust
use chunker::chunking::{chunk_data, ChunkStream};
use std::io::BufReader;

// Eager chunking
let data = b"data to chunk";
let chunks = chunk_data(data, None, None, None)?;
println!("First chunk hash (hex): {}", chunks[0].hash_hex());

// Streaming chunking
let file = BufReader::new(std::fs::File::open("/path/to/large.nar")?);
for chunk in ChunkStream::new(file, None, None, None) {
    let chunk = chunk?;
    println!("Chunk at {} ({} bytes)", chunk.offset, chunk.length);
}
```

## Build & test
```bash
cargo build           # debug
cargo test            # run tests
cargo build --release # optimized
```

## Async notes
- `chunk_data_async` collects chunk metadata in memory and enforces a buffer cap (default 2 GiB, tunable via `CHUNKER_ASYNC_BUFFER_LIMIT_BYTES`, clamped to 64 MiB–3 GiB). For very large inputs, prefer streaming chunkers.
- `chunk_stream_blocking_adapter` bridges async readers via blocking reads; call from a blocking task.
- `chunk_stream_async` offloads the blocking work to a thread so it won’t stall your async runtime.

## Benchmark snapshots (single-thread)

| Path                    | Size / Params                        | Throughput      | Notes                                   |
| ----------------------- | ------------------------------------ | --------------- | --------------------------------------- |
| FastCDC raw             | 10 MiB (256K/1M/4M)                  | ~2.28 GiB/s     | Reference chunker only                  |
| ChunkStream (hash+copy) | 10 MiB (defaults)                    | ~282 MiB/s      | End-to-end chunk+hash (SHA-256)         |
| Eager chunk descriptors | 10 MiB (defaults)                    | ~634 MiB/s      | Hash/offset/length only                 |
| SHA-256 hash            | 1 MiB                                | ~1.61 GiB/s     | Default compatibility hash              |
| BLAKE3 hash             | 1 MiB                                | ~4.03 GiB/s     | Explicit opt-in hash                    |
| Zstd compress           | 1 MiB zeros, level 3                 | ~4.38 GiB/s     | With buffer reuse                       |

CPU note: numbers are from `benchmarks/latest.md` generated with `RUSTFLAGS="-C target-cpu=native"` and a release benchmark build. Rerun `just bench` or `cargo bench` on your hardware (x86_64/ARM) to get local figures.

## Observability

`chunker` emits `tracing` events and metrics from the library hot paths. Applications that need OTLP, Jaeger, Datadog, Honeycomb, or another backend should install the exporter/subscriber at the binary or host application boundary.

For Elixir, use the NIF `enable_logging/1` helper for runtime debugging and wire distributed tracing in the BEAM application.

## License
Apache-2.0
