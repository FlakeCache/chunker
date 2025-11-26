# Chunker

High-performance content-defined chunking (FastCDC) for Nix NARs.

**Status**: Internal FlakeCache project | Nix archive handler for production binary caches

## Features

- **FastCDC Chunking**: Content-defined chunking algorithm for optimal deduplication
- **QuickCDC (feature flag)**: Lightweight gear-hash chunker with fixed-mask style boundaries
- **Multiple Compression**: zstd, xz (LZMA2), bzip2 support
- **Cryptographic Signing**: Ed25519 signatures for data authenticity
- **Hash Computation**: SHA256 and Nix base32 encoding
- **Rustler NIF**: Native Elixir bindings for high performance
- **Standalone Library**: Use as a Rust library independent of Elixir

## Quick Start

### As Elixir Rustler NIF

```elixir
alias FlakecacheApp.Native.Chunker

# Hash computation
hash = Chunker.sha256_hash(data)

# Content-defined chunking with deduplication (strategy selectable)
{:ok, chunks} = Chunker.chunk_data(data, 16_384, 65_536, 262_144, "fastcdc")

# Compression/decompression
{:ok, compressed} = Chunker.compress_zstd(data, 3)
{:ok, original} = Chunker.decompress_zstd(compressed)

# Ed25519 signing
{:ok, {secret_key, public_key}} = Chunker.generate_keypair()
{:ok, signature} = Chunker.sign_data(data, secret_key)
:ok = Chunker.verify_signature(data, signature, public_key)
```

### As Rust Library

```rust
use chunker::chunking::{
    chunk_boundaries_with_strategy, ChunkingStrategyKind,
};

let data = b"data to chunk";
let chunks = chunk_boundaries_with_strategy(
    data,
    Some(16_384),
    Some(65_536),
    Some(262_144),
    ChunkingStrategyKind::FastCdc,
)?;
for chunk in chunks {
    println!("Chunk at {}: {} bytes", chunk.offset, chunk.length);
}
```

## Building

### Prerequisites

- Rust 1.70+
- Cargo
- (Optional for Elixir NIF) Elixir 1.14+, Rustler plugin

### Development

```bash
cargo build
cargo test
```

### Release (optimized)

```bash
cargo build --release
```

### Testing

```bash
# All tests (unit + Rust)
cargo test

# Only unit tests
cargo test --lib

# With coverage
cargo tarpaulin --out Html --output-dir cover/
```

## Module Structure

- **`chunking.rs`** - FastCDC chunking implementation
- **`compression.rs`** - zstd/xz/bzip2 compression handlers
- **`hashing.rs`** - SHA256 and Nix base32 encoding
- **`signing.rs`** - Ed25519 keypair and signature operations
- **`lib.rs`** - Rustler NIF initialization

## Performance Notes

FastCDC provides content-aware chunking:

- **Deduplication**: Identical content → identical chunks (position-independent)
- **Resilience**: Content insertion/deletion doesn't affect unrelated chunks
- **Configurability**: Adjustable min (16KB), avg (64KB), max (256KB) sizes

Typical speeds on modern hardware:

- Chunking: ~500 MB/s
- SHA256: ~1 GB/s (hardware-accelerated)
- Compression: 50-300 MB/s (algorithm-dependent)

## Versioning & Releases

Releases published to [GitHub Releases](https://github.com/FlakeCache/chunker/releases) (private).

Format: `MAJOR.MINOR.PATCH` (semver)
- Pre-releases: `-alpha`, `-beta`, `-rc`

## License

Apache License 2.0
