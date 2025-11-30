# Requirements Document: World-Class Performance for Chunker Library

## Introduction

This requirements document specifies performance optimizations for the Chunker library, a high-performance content-defined chunking system for Nix NARs written in Rust. The library must handle hundreds of millions of chunking operations where every millisecond matters.

The current implementation includes:
- FastCDC content-defined chunking with parallel hashing via Rayon
- Compression codecs (zstd, xz, lz4, bzip2) with scratch buffer reuse
- SHA256/BLAKE3 hashing with Nix base32 encoding
- Ed25519 signing with zeroization
- Elixir NIF bindings via Rustler

This document defines requirements across seven focus areas: streaming/pipeline efficiency, concurrency/parallelism, I/O optimization, NIF performance, compile-time optimizations, observability, and API consistency.

---

## Requirements

### Requirement 1: Pipeline Parallelism Architecture

**User Story:** As a system operator processing Nix NARs, I want chunking, hashing, and compression to run in parallel pipeline stages, so that I maximize CPU utilization and throughput.

#### Acceptance Criteria

1. WHEN a streaming chunker is processing data THEN the system SHALL support concurrent execution of chunking, hashing, and compression stages in a pipelined architecture.

2. WHEN pipeline parallelism is enabled THEN the system SHALL allow chunk N+1 to begin FastCDC processing WHILE chunk N is being hashed AND chunk N-1 is being compressed.

3. WHEN pipeline stages have different throughput rates THEN the system SHALL implement bounded channels between stages with configurable capacity (default: 2x the number of CPU cores).

4. IF a downstream stage (compression) is slower than upstream (chunking) THEN the system SHALL apply backpressure by blocking the upstream producer when the inter-stage buffer reaches capacity.

5. WHEN configuring pipeline parallelism THEN the system SHALL provide a builder API that allows enabling/disabling individual pipeline stages:
   - `PipelineBuilder::new().with_hashing(true).with_compression(Some(CompressionStrategy::Balanced)).build()`

6. WHEN pipeline parallelism is disabled THEN the system SHALL fall back to the current sequential-with-parallel-hashing model for backward compatibility.

---

### Requirement 2: Async Streaming Optimizations

**User Story:** As a developer integrating the chunker with async runtimes, I want true non-blocking async streaming, so that I can efficiently process data without blocking executor threads.

#### Acceptance Criteria

1. WHEN using `ChunkStreamAsync` THEN the system SHALL use native async I/O without `block_on` bridges that can block the executor thread.

2. WHEN the `async-stream` feature is enabled THEN the system SHALL provide a `ChunkStreamAsync` implementation that is compatible with Tokio, async-std, and smol runtimes.

3. IF an async reader yields `Poll::Pending` THEN the system SHALL properly suspend the stream future and resume when data becomes available WITHOUT busy-waiting.

4. WHEN processing async streams THEN the system SHALL support zero-copy chunk emission using `Bytes` handles that reference the underlying buffer.

5. WHEN the async buffer limit is reached (configurable via `CHUNKER_ASYNC_BUFFER_LIMIT_BYTES`) THEN the system SHALL return `ChunkingError::BufferLimitExceeded` rather than causing an out-of-memory condition.

6. WHEN async streaming is used THEN the system SHALL provide a `poll_next_chunk()` method for manual polling integration with custom executors.

---

### Requirement 3: Adaptive Buffer Sizing

**User Story:** As a system operator, I want the chunker to automatically tune buffer sizes based on workload characteristics, so that I achieve optimal throughput without manual configuration.

#### Acceptance Criteria

1. WHEN initializing a `ChunkStream` THEN the system SHALL start with a conservative buffer allocation (min of `min_chunk_size` and 64KB) and grow dynamically.

2. WHEN buffer growth is required THEN the system SHALL use exponential growth (2x) up to the configured read slice cap (`CHUNKER_READ_SLICE_CAP_BYTES`, default 16MB).

3. WHEN processing a sequence of chunks THEN the system SHALL track the rolling average chunk size and pre-allocate buffers based on observed patterns.

4. IF the workload exhibits consistent chunk sizes THEN the system SHALL stabilize buffer allocation to avoid repeated resize operations.

5. WHEN memory pressure is detected (optional, via feature flag) THEN the system SHALL reduce buffer sizes and accept lower throughput to prevent OOM conditions.

6. WHEN configuring buffer behavior THEN the system SHALL provide explicit options:
   - `BufferStrategy::Fixed(size)` - Use fixed-size buffers
   - `BufferStrategy::Adaptive { min, max, growth_factor }` - Dynamic sizing
   - `BufferStrategy::PreAllocated(size)` - Single upfront allocation

---

### Requirement 4: Rayon Thread Pool Optimization

**User Story:** As a system administrator, I want fine-grained control over Rayon's thread pool configuration, so that I can tune parallelism for my specific hardware and workload.

#### Acceptance Criteria

1. WHEN the chunker library initializes THEN the system SHALL respect `RAYON_NUM_THREADS` environment variable for thread pool sizing.

2. WHEN no environment variable is set THEN the system SHALL default to `num_cpus::get()` threads for hashing workloads.

3. WHEN hashing small batches (fewer than 4 chunks) THEN the system SHALL process sequentially to avoid Rayon overhead, as currently implemented.

4. WHEN configuring the thread pool programmatically THEN the system SHALL provide:
   - `ChunkerConfig::with_thread_pool(rayon::ThreadPool)` for custom pool injection
   - `ChunkerConfig::with_parallelism_threshold(usize)` to set the sequential/parallel cutoff

5. WHEN processing chunks in parallel THEN the system SHALL use work-stealing scheduling to balance load across cores with varying chunk sizes.

6. WHEN the chunker is used as a library in multi-tenant applications THEN the system SHALL support isolated thread pools per chunker instance to prevent interference.

---

### Requirement 5: Lock-Free Metrics Collection

**User Story:** As an operator monitoring production systems, I want metrics collection to have near-zero overhead, so that observability does not impact chunking throughput.

#### Acceptance Criteria

1. WHEN the `metrics` feature is enabled THEN the system SHALL use atomic counters for `chunker.chunks_emitted` and `chunker.bytes_processed`.

2. WHEN updating histograms (`chunker.chunk_size`) in hot paths THEN the system SHALL use lock-free histogram implementations (e.g., HDR histogram with atomic updates).

3. IF metrics recording would require a lock THEN the system SHALL use thread-local aggregation with periodic merging (configurable interval, default 100ms).

4. WHEN metrics are disabled at compile time THEN the system SHALL produce zero runtime overhead through dead code elimination.

5. WHEN the sampling ratio is configured THEN the system SHALL support probabilistic sampling for high-frequency metrics (e.g., record 1 in N chunk sizes).

6. WHEN exporting metrics THEN the system SHALL support both push (Prometheus Pushgateway) and pull (scrape endpoint) models.

---

### Requirement 6: I/O Read Buffer Optimization

**User Story:** As a developer processing large NAR files, I want optimized I/O patterns, so that disk and network reads do not bottleneck chunking throughput.

#### Acceptance Criteria

1. WHEN reading from files THEN the system SHALL use read buffer sizes that are multiples of the filesystem block size (typically 4KB).

2. WHEN the `io_uring` feature is enabled (Linux 5.1+) THEN the system SHALL use asynchronous I/O via io_uring for file reads.

3. WHEN processing memory-mapped files THEN the system SHALL support `mmap`-based readers via:
   - `ChunkStream::from_mmap(memmap2::Mmap)` for read-only access
   - Automatic fallback to buffered reads if mmap fails

4. WHEN configuring read behavior THEN the system SHALL provide:
   - `CHUNKER_READ_SLICE_CAP_BYTES` for maximum per-read size (current: 16MB)
   - `CHUNKER_READAHEAD_BYTES` for OS-level readahead hints

5. IF the data source supports vectored I/O THEN the system SHALL use `readv()` to minimize syscall overhead when filling the internal buffer.

6. WHEN using direct I/O (`O_DIRECT`) THEN the system SHALL ensure buffers are page-aligned and handle partial reads correctly.

---

### Requirement 7: Memory-Mapped File Support

**User Story:** As an operator with high-memory systems, I want to use memory-mapped file access for large NARs, so that I can leverage OS page cache and reduce copy overhead.

#### Acceptance Criteria

1. WHEN a file path is provided THEN the system SHALL offer `ChunkStream::open_mmap(path)` that memory-maps the file read-only.

2. WHEN using mmap THEN the system SHALL pass the mapped memory directly to FastCDC without intermediate copies.

3. IF mmap initialization fails (e.g., file too large for address space on 32-bit) THEN the system SHALL automatically fall back to buffered I/O with a warning log.

4. WHEN the mmap feature is used THEN the system SHALL call `madvise(MADV_SEQUENTIAL)` to hint sequential access patterns to the kernel.

5. WHEN the chunker completes or is dropped THEN the system SHALL properly unmap the file to release virtual address space.

6. WHEN mmap is used with very large files (>1GB) THEN the system SHALL support windowed/sliding mmap to avoid exhausting virtual address space.

---

### Requirement 8: NIF DirtyCpu Scheduler Optimization

**User Story:** As an Elixir developer, I want NIF calls to efficiently use Dirty CPU schedulers, so that long-running chunking operations do not block the BEAM scheduler.

#### Acceptance Criteria

1. WHEN a NIF function performs CPU-intensive work (chunking, compression, hashing) THEN the function SHALL be annotated with `#[rustler::nif(schedule = "DirtyCpu")]`.

2. WHEN data is passed from Elixir to the NIF THEN the system SHALL use Rustler's `Binary` type to achieve zero-copy access to the BEAM binary.

3. WHEN returning data to Elixir THEN the system SHALL use `OwnedBinary` with pre-allocated capacity to minimize allocations:
   - Estimate output size based on input size and compression ratio
   - Use `OwnedBinary::new(estimated_size)` rather than growing dynamically

4. IF a NIF operation is expected to take longer than 1ms THEN the system SHALL yield periodically using Rustler's yielding mechanisms (if available) or split work across multiple NIF calls.

5. WHEN batch processing multiple chunks THEN the system SHALL provide batch NIF APIs:
   - `chunk_data_batch(list_of_binaries)` to amortize NIF call overhead
   - `compress_batch(list_of_binaries, strategy)` for bulk compression

6. WHEN errors occur in NIF code THEN the system SHALL return Erlang atoms (e.g., `:io_error`, `:invalid_chunking_options`) rather than raising exceptions.

---

### Requirement 9: NIF Binary Handling Efficiency

**User Story:** As an Elixir developer processing large binaries, I want minimal copying between Elixir and Rust, so that I avoid doubling memory usage for large NARs.

#### Acceptance Criteria

1. WHEN receiving Elixir binaries in NIFs THEN the system SHALL use `Binary<'a>` lifetime-bound references that directly access BEAM memory.

2. WHEN the NIF needs to modify binary data THEN the system SHALL copy to an `OwnedBinary` only when mutation is required.

3. WHEN returning chunk metadata to Elixir THEN the system SHALL return tuples `{hash_hex, offset, length}` without including payload data unless explicitly requested.

4. IF the caller needs chunk payloads THEN the system SHALL provide `chunk_data_with_payloads/4` that returns `{hash, offset, binary_slice}` using sub-binary references.

5. WHEN processing streaming data from Elixir THEN the system SHALL support a resource-based API:
   - `chunker_open(options) -> resource`
   - `chunker_feed(resource, binary) -> {:ok, [chunks]} | {:more}`
   - `chunker_finish(resource) -> {:ok, [final_chunks]}`

6. WHEN term conversion is required THEN the system SHALL use efficient encoder/decoder implementations that avoid intermediate string allocations.

---

### Requirement 10: LTO and Codegen Optimization

**User Story:** As a release engineer, I want optimal compiler settings for production builds, so that the library achieves maximum runtime performance.

#### Acceptance Criteria

1. WHEN building release binaries THEN the system SHALL use Link-Time Optimization (LTO) with `lto = true` in Cargo.toml (currently implemented).

2. WHEN building release binaries THEN the system SHALL use single codegen unit (`codegen-units = 1`) for maximum optimization (currently implemented).

3. WHEN the `asm` feature is enabled THEN the system SHALL use assembly-optimized SHA256 via `sha2/asm` (currently implemented).

4. WHEN targeting specific CPU architectures THEN the build system SHALL support:
   - `RUSTFLAGS="-C target-cpu=native"` for local builds
   - Pre-built binaries for common targets (x86_64-v3, aarch64)

5. WHEN building for production THEN the system SHALL strip debug symbols (`strip = true`) and disable debug info (`debug = 0`) (currently implemented).

6. WHEN PGO (Profile-Guided Optimization) is available THEN the build system SHALL document the process:
   - Instrumented build for profiling
   - Benchmark suite execution
   - Optimized build using profile data

---

### Requirement 11: Profile-Guided Optimization Support

**User Story:** As a performance engineer, I want to use PGO for production builds, so that the compiler can optimize based on real-world usage patterns.

#### Acceptance Criteria

1. WHEN PGO is requested THEN the build system SHALL provide a documented workflow:
   ```
   cargo build --release --features pgo-instrument
   ./run_benchmarks.sh  # Generates .profdata
   cargo build --release --features pgo-optimize
   ```

2. WHEN generating PGO profiles THEN the system SHALL include a representative benchmark suite covering:
   - Various chunk sizes (small: 64KB, medium: 1MB, large: 4MB)
   - Different compression strategies (Fastest, Balanced, Smallest)
   - Both SHA256 and BLAKE3 hashing

3. WHEN PGO is enabled THEN the system SHALL verify performance improvements via automated benchmarks comparing PGO vs non-PGO builds.

4. IF PGO profiles are incompatible with compiler version THEN the build system SHALL fail with a clear error message recommending profile regeneration.

5. WHEN distributing pre-built binaries THEN the system SHALL document whether PGO was used and with what workload profile.

---

### Requirement 12: Feature Flag Optimization

**User Story:** As a library consumer, I want to enable only the features I need, so that I minimize binary size and compile time.

#### Acceptance Criteria

1. WHEN compiling with default features THEN the system SHALL produce a minimal build with:
   - FastCDC chunking
   - SHA256 and BLAKE3 hashing
   - Assembly-optimized crypto (`asm` feature)

2. WHEN the `nif` feature is enabled THEN the system SHALL include Rustler bindings and produce a cdylib.

3. WHEN the `async-stream` feature is enabled THEN the system SHALL include `ChunkStreamAsync` and futures-based APIs.

4. WHEN the `telemetry` feature is enabled THEN the system SHALL include OpenTelemetry integration (mutually exclusive with `nif` as documented).

5. WHEN features are combined THEN the system SHALL validate compatible combinations at compile time using `#[cfg]` attributes.

6. WHEN documenting the library THEN the system SHALL clearly specify feature dependencies and binary size impact for each feature combination.

---

### Requirement 13: Zero-Cost Tracing

**User Story:** As an operator, I want tracing to be completely eliminated when disabled, so that production builds have zero observability overhead.

#### Acceptance Criteria

1. WHEN tracing is disabled at compile time THEN all `trace!`, `debug!`, and `instrument` macros SHALL compile to no-ops with zero runtime cost.

2. WHEN tracing is enabled but the log level is above the event level THEN the system SHALL perform level checking before any argument evaluation.

3. WHEN the current implementation's sampling strategy is used (1 in 1024 traces) THEN the system SHALL use atomic counter increments that do not require locks.

4. WHEN configuring runtime tracing levels THEN the system SHALL support dynamic level changes via `tracing_subscriber::reload`.

5. WHEN tracing is enabled in production THEN the system SHALL support structured logging with:
   - Span context propagation for distributed tracing
   - Efficient JSON formatting without allocation for common fields

6. WHEN the `telemetry` feature is enabled THEN the system SHALL export spans to OpenTelemetry-compatible backends (Jaeger, Datadog, Honeycomb).

---

### Requirement 14: Sampling Strategies for High-Volume Metrics

**User Story:** As an operator with extremely high throughput, I want configurable sampling for metrics and traces, so that observability overhead remains bounded.

#### Acceptance Criteria

1. WHEN the sampling rate is configured THEN the system SHALL support:
   - Fixed-rate sampling: 1 in N (e.g., 1 in 1000)
   - Adaptive sampling: Higher rate for errors/anomalies

2. WHEN sampled metrics are exported THEN the system SHALL include sample rate metadata for accurate aggregation.

3. WHEN trace sampling is enabled THEN the system SHALL support head-based sampling (decide at span start) for consistent trace completeness.

4. IF a parent span is sampled THEN all child spans SHALL also be sampled to maintain trace integrity.

5. WHEN configuring sampling THEN the system SHALL provide environment variables:
   - `CHUNKER_TRACE_SAMPLE_RATE=0.001` (0.1% of traces)
   - `CHUNKER_METRICS_SAMPLE_RATE=0.01` (1% of metric events)

6. WHEN the sampling rate is 0 THEN the system SHALL completely bypass metric/trace recording logic.

---

### Requirement 15: API Consistency - Missing _into() Variants

**User Story:** As a library user optimizing for allocation, I want consistent `_into()` variants for all compression functions, so that I can reuse buffers across operations.

#### Acceptance Criteria

1. WHEN using bzip2 compression THEN the system SHALL provide `compress_bzip2_into(data, output)` and `compress_bzip2_into(data, level, output)`.

2. WHEN using XZ compression THEN the system SHALL provide `compress_xz_into(data, output)` and `compress_xz_into(data, level, output)`.

3. WHEN `_into()` variants are called THEN the output buffer SHALL be cleared before writing new data (consistent with existing `compress_zstd_into` behavior).

4. WHEN using `CompressionScratch` THEN all compression strategies SHALL use `_into()` variants internally:
   - `CompressionStrategy::Smallest` currently falls back to allocating; it SHALL use `compress_xz_into`.

5. WHEN documenting the API THEN all compression functions SHALL follow the pattern:
   - `compress_<algo>(data) -> Result<Vec<u8>>` - Allocating version
   - `compress_<algo>_into(data, output) -> Result<()>` - Buffer-reusing version

6. WHEN using `_into()` variants THEN the system SHALL return the number of bytes written or provide access via `output.len()` after the call.

---

### Requirement 16: Consistent Error Handling

**User Story:** As a library consumer, I want consistent error types and handling across all modules, so that I can write robust error-handling code.

#### Acceptance Criteria

1. WHEN errors occur in any module THEN the system SHALL return typed errors:
   - `ChunkingError` for chunking operations
   - `CompressionError` for compression operations
   - `HashingError` for hashing operations
   - `SigningError` for signing operations

2. WHEN errors contain context THEN the system SHALL use structured error types with fields (e.g., `ChunkingError::Bounds { data_len, offset, length }`).

3. WHEN errors are displayed THEN the system SHALL produce machine-parseable messages in lowercase_snake_case format (e.g., `bounds_check_failed`).

4. WHEN NIF functions encounter errors THEN the system SHALL return Erlang atoms rather than string messages for efficient pattern matching.

5. WHEN using the `thiserror` crate THEN all error types SHALL implement `std::error::Error` and `Display`.

6. WHEN errors need to be converted between modules THEN the system SHALL provide `From` implementations for common conversions (e.g., `std::io::Error` to `ChunkingError::Io`).

---

### Requirement 17: Batch Processing APIs

**User Story:** As a developer processing many small items, I want batch APIs that amortize per-call overhead, so that I achieve higher throughput for small workloads.

#### Acceptance Criteria

1. WHEN processing multiple chunks for hashing THEN the system SHALL provide `hash_batch(chunks: &[&[u8]]) -> Vec<[u8; 32]>`.

2. WHEN processing multiple items for compression THEN the system SHALL provide `compress_batch(items: &[&[u8]], strategy) -> Vec<Vec<u8>>`.

3. WHEN using batch APIs THEN the system SHALL process items in parallel using Rayon when the batch size exceeds the parallelism threshold.

4. WHEN batch processing fails for a single item THEN the system SHALL return `Result<Vec<Result<T, E>>, E>` allowing partial success handling.

5. WHEN using NIF batch APIs THEN the system SHALL accept Elixir lists and return lists, minimizing NIF call overhead for many small operations.

6. WHEN batch sizes are very large THEN the system SHALL process in chunks to avoid excessive memory usage and maintain progress visibility.

---

### Requirement 18: Work-Stealing Efficiency

**User Story:** As a developer processing variable-size chunks, I want efficient load balancing across cores, so that I maximize CPU utilization regardless of chunk size distribution.

#### Acceptance Criteria

1. WHEN processing chunks in parallel THEN the system SHALL use Rayon's work-stealing scheduler that dynamically balances work.

2. WHEN chunks have highly variable sizes THEN the system SHALL consider chunk size when scheduling to avoid assigning all large chunks to a single thread.

3. WHEN processing streaming data THEN the system SHALL emit chunks to worker threads as they become available rather than waiting for a full batch.

4. IF work-stealing overhead exceeds benefit (very small chunks) THEN the system SHALL automatically fall back to sequential processing.

5. WHEN monitoring parallel efficiency THEN the system SHALL provide optional metrics:
   - `chunker.parallel_efficiency` (work time / wall time)
   - `chunker.work_stealing_events` (number of steals)

6. WHEN configuring work distribution THEN the system SHALL support custom partitioning strategies via trait objects.

---

## Non-Functional Requirements

### NFR 1: Throughput Targets

1. WHEN processing in-memory data with BLAKE3 hashing THEN the system SHALL achieve at least 2 GB/s throughput on a modern 8-core CPU.

2. WHEN processing in-memory data with SHA256 hashing (asm-enabled) THEN the system SHALL achieve at least 1 GB/s throughput on a modern 8-core CPU.

3. WHEN compression is enabled (zstd level 3) THEN the system SHALL achieve at least 500 MB/s throughput on a modern 8-core CPU.

### NFR 2: Latency Requirements

1. WHEN processing a single 1MB chunk THEN the system SHALL complete chunking + hashing in under 1ms (p99).

2. WHEN the NIF is called from Elixir THEN the overhead of the NIF boundary crossing SHALL be under 10 microseconds.

### NFR 3: Memory Efficiency

1. WHEN streaming large files THEN the system SHALL use no more than 2x the configured buffer limit in peak memory.

2. WHEN using `CompressionScratch` THEN the system SHALL reuse allocations and not grow unboundedly across calls.

### NFR 4: Compatibility

1. WHEN compiled as a library THEN the system SHALL support Rust 2024 edition and maintain MSRV of Rust 1.85+.

2. WHEN compiled as a NIF THEN the system SHALL be compatible with Erlang/OTP 25+ and Elixir 1.15+.

### NFR 5: Security

1. WHEN handling cryptographic keys THEN the system SHALL zeroize sensitive data after use (currently implemented for Ed25519 signing).

2. WHEN decompressing data THEN the system SHALL enforce size limits to prevent decompression bomb attacks (currently implemented).

---

## Glossary

- **FastCDC**: Fast Content-Defined Chunking algorithm that finds chunk boundaries based on data content
- **NAR**: Nix ARchive format used by the Nix package manager
- **NIF**: Native Implemented Function - Erlang's FFI mechanism
- **LTO**: Link-Time Optimization - compiler optimization across compilation units
- **PGO**: Profile-Guided Optimization - optimization based on runtime profiling data
- **Rayon**: Rust library for data parallelism with work-stealing
- **DirtyCpu**: BEAM scheduler type for long-running CPU-bound operations
- **BLAKE3**: Modern cryptographic hash function optimized for speed
