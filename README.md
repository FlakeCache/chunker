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
- `chunk_data_async` enforces a 512 MiB buffer cap; reject larger streams to avoid OOM.
- `chunk_stream_blocking` bridges async readers via blocking reads; call from a blocking task.
- `chunk_stream_async` offloads the blocking work to a thread so it won’t stall your async runtime.

## Benchmark snapshots (single-thread)
| Path                    | Size / Params                        | Throughput      | Notes                                   |
| ----------------------- | ------------------------------------ | --------------- | --------------------------------------- |
| FastCDC raw             | 10 MiB (256K/1M/4M)                  | ~2.0 GiB/s      | Reference chunker only                  |
| ChunkStream (hash+copy) | 10 MiB (defaults)                    | ~0.95–1.1 GiB/s | End-to-end chunk+hash (SHA-256)         |
| SHA-256 hash            | 1 MiB                                | ~2.0 GiB/s      | Hardware-accelerated                    |
| Zstd compress           | 1 MiB zeros, level 3                 | ~4.6 GiB/s      | With buffer reuse                       |

CPU note: numbers are from a recent desktop CPU with `RUSTFLAGS="-C target-cpu=native"` and a release build. Rerun `cargo bench` on your hardware (x86_64/ARM) to get local figures.

## Telemetry & Observability

By default, `chunker` uses standard logging (via `tracing`) which prints to stderr. This is ideal for CLI usage and simple debugging.

For production deployments or performance analysis, you can enable the `telemetry` feature. This switches the binary to use:
- **Tokio Runtime**: For async telemetry export.
- **OTLP Exporter**: Sends traces to Jaeger, Honeycomb, or any OpenTelemetry collector.
- **Async Pipeline**: Ensures telemetry doesn't block the main chunking loop.

**When to use `telemetry`:**
- **CLI Binary**: Enable for production/profiling (`cargo build --release --features telemetry`).
- **Build Runners / CI**: Perfect for monitoring chunking performance in CI pipelines.
- **Elixir NIF**: **Do not enable**. The NIF uses `tracing` but relies on the host VM or simple logging. Enabling the `telemetry` feature (and its Tokio runtime) inside a NIF is not recommended.

```bash
# Build with full telemetry support (CLI only)
cargo build --release --features telemetry

# Run with a local Jaeger instance
docker run -d -p 4317:4317 -p 16686:16686 jaegertracing/all-in-one:latest
./target/release/chunker large_file.bin

# Pipeline usage (reading from stdin)
# Useful for streaming data from S3 or build artifacts directly
aws s3 cp s3://bucket/file.tar.gz - | ./target/release/chunker -
```

## License
Apache-2.0
