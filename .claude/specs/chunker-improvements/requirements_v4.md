# Requirements Document: World-Class Chunker Performance

## Introduction

This document defines requirements for transforming the Chunker library into a world-class, high-performance content-defined chunking solution optimized for processing hundreds of millions of NAR files. The library currently provides FastCDC chunking, multiple compression codecs (zstd, xz, lz4, bzip2), cryptographic hashing (SHA256, BLAKE3), Ed25519 signing, and Elixir NIF bindings.

The focus areas include NAR-specific optimizations, intelligent caching strategies, batch processing capabilities, resource management, storage backend integration, comprehensive monitoring, and future-proofing for hardware acceleration and new algorithms.

**Performance Target**: Every millisecond matters at scale of 100s of millions of operations.

---

## Requirements

### Requirement 1: NAR File Structure Analysis

**User Story:** As a system architect, I want the chunker to understand NAR file structure, so that chunk boundaries align with NAR semantics for better deduplication and cache efficiency.

#### Acceptance Criteria

1. WHEN processing a NAR file THEN the system SHALL parse the NAR magic bytes (`\x0d\x00\x00\x00\x00\x00\x00\x00nix-archive-1\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00`) to identify NAR format.
2. WHEN a NAR header is detected THEN the system SHALL provide an option to emit the header as a separate chunk to maximize header deduplication across archives.
3. WHEN chunking NAR files THEN the system SHALL expose a `NarAwareChunker` mode that biases chunk boundaries toward NAR entry boundaries (file/directory markers).
4. IF NAR-aware mode is enabled AND a file entry boundary is within `avg_size / 4` of a natural FastCDC cut point THEN the system SHALL prefer the entry boundary as the cut point.
5. WHEN NAR-aware chunking is used THEN the system SHALL maintain metadata mapping chunks to original NAR entries for reconstruction validation.
6. WHEN processing large NAR files (>100MB) THEN the system SHALL support streaming NAR parsing without buffering the entire archive.

---

### Requirement 2: Optimal Chunk Boundaries for NAR Format

**User Story:** As a storage engineer, I want chunk boundaries optimized for NAR deduplication patterns, so that similar derivations share maximum chunks.

#### Acceptance Criteria

1. WHEN configuring the chunker THEN the system SHALL provide NAR-optimized default chunk sizes: `min_size=128KB`, `avg_size=512KB`, `max_size=2MB` (tuned for typical Nix package sizes).
2. WHEN a chunk contains only padding bytes (NAR uses 8-byte alignment padding) THEN the system SHALL NOT emit standalone padding-only chunks but merge them with adjacent content.
3. WHEN processing executable files within NAR THEN the system SHALL support an optional mode to detect ELF/Mach-O section boundaries for deduplication of shared libraries.
4. IF the `dedup_hint` flag is enabled THEN the system SHALL compute a rolling fingerprint that prefers cut points at instruction boundaries for binary content.
5. WHEN the same file content appears in multiple NARs THEN chunk boundaries SHALL be deterministic and reproducible given identical chunking parameters.

---

### Requirement 3: Chunk Hash Caching

**User Story:** As an operations engineer, I want chunk hashes cached in memory, so that repeated chunking of identical content avoids redundant hash computation.

#### Acceptance Criteria

1. WHEN the `HashCache` feature is enabled THEN the system SHALL maintain an LRU cache mapping content fingerprints to precomputed hashes.
2. WHEN a chunk's rolling hash fingerprint matches a cached entry THEN the system SHALL verify via full content comparison before returning the cached hash (collision protection).
3. WHEN the cache reaches its configured `max_entries` limit THEN the system SHALL evict least-recently-used entries.
4. WHEN the cache reaches its configured `max_memory_bytes` limit THEN the system SHALL evict entries until under budget.
5. IF cache statistics are requested THEN the system SHALL report: `hit_count`, `miss_count`, `eviction_count`, `memory_usage_bytes`, `hit_rate_percent`.
6. WHEN multi-threaded chunking is active THEN the cache SHALL use a concurrent data structure (e.g., `DashMap`) to avoid lock contention.
7. WHEN a cache entry is older than `ttl_seconds` (configurable, default 3600) THEN the system SHALL treat it as a miss and recompute.

---

### Requirement 4: Compression Dictionary Caching

**User Story:** As a performance engineer, I want zstd dictionaries cached and reused, so that dictionary training costs are amortized across millions of compressions.

#### Acceptance Criteria

1. WHEN `DictionaryCache` is enabled THEN the system SHALL cache trained `EncoderDictionary` and `DecoderDictionary` objects by dictionary ID.
2. WHEN compressing with a dictionary ID THEN the system SHALL first check the cache before loading/training.
3. WHEN the dictionary cache reaches `max_dictionaries` (default 16) THEN the system SHALL evict least-recently-used dictionaries.
4. WHEN a dictionary is loaded from bytes THEN the system SHALL cache both encoder and decoder variants to avoid redundant parsing.
5. IF dictionary training is requested THEN the system SHALL provide an async API that trains on sample data without blocking the chunking pipeline.
6. WHEN using dictionary compression THEN the system SHALL include the dictionary ID in chunk metadata for decompression routing.

---

### Requirement 5: Memoization Opportunities

**User Story:** As a developer, I want expensive computations memoized, so that hot paths avoid redundant work.

#### Acceptance Criteria

1. WHEN `ChunkingOptions::resolve()` is called with identical parameters THEN the system SHALL return a cached validated options struct (avoid repeated validation).
2. WHEN the same data slice is hashed multiple times within a single chunking operation THEN the system SHALL detect and reuse the hash result.
3. WHEN compression scratch buffers are used THEN the system SHALL provide a `ScratchPool` that recycles allocations across operations.
4. WHEN NIF bindings process the same binary multiple times THEN the system SHALL support caller-provided caching hints to skip redundant work.

---

### Requirement 6: Multi-File Batch Chunking API

**User Story:** As a pipeline developer, I want to chunk multiple files in a single API call, so that setup costs are amortized and parallelism is maximized.

#### Acceptance Criteria

1. WHEN `batch_chunk_files(paths: &[PathBuf], options: ChunkingOptions)` is called THEN the system SHALL process files in parallel using the Rayon thread pool.
2. WHEN batch processing THEN the system SHALL emit results in a streaming fashion (channel or iterator) rather than waiting for all files.
3. IF a file in the batch fails THEN the system SHALL continue processing other files and report the failure in the result stream.
4. WHEN batch processing THEN the system SHALL reuse chunking state (FastCDC gear table, hash contexts) across files.
5. WHEN batch processing NAR files THEN the system SHALL optionally deduplicate identical chunks across the batch before returning.
6. WHEN processing >100 files THEN the system SHALL report progress metrics: `files_completed`, `files_total`, `bytes_processed`, `elapsed_time`.

---

### Requirement 7: Batch Compression API

**User Story:** As a storage engineer, I want to compress multiple chunks in a batch, so that compression context setup is amortized.

#### Acceptance Criteria

1. WHEN `batch_compress(chunks: &[Bytes], strategy: CompressionStrategy)` is called THEN the system SHALL compress chunks in parallel.
2. WHEN using zstd with a shared dictionary THEN the system SHALL initialize the dictionary once and share across compression workers.
3. WHEN batch compression uses `CompressionScratch` THEN the system SHALL maintain a pool of scratch buffers (one per thread).
4. WHEN compressing chunks of similar size THEN the system SHALL pre-allocate output buffers based on historical compression ratios.
5. IF compression of any chunk fails THEN the system SHALL return the error alongside successful results (partial success model).
6. WHEN batch size exceeds `parallel_threshold` (default 8) THEN the system SHALL use parallel execution; below threshold, use sequential.

---

### Requirement 8: Vectorized Operations

**User Story:** As a performance engineer, I want SIMD-optimized operations, so that CPU throughput is maximized.

#### Acceptance Criteria

1. WHEN BLAKE3 hashing is used THEN the system SHALL leverage the `blake3` crate's built-in SIMD and multi-threading support.
2. WHEN SHA256 hashing is used with the `asm` feature THEN the system SHALL use SHA-NI instructions on supported CPUs.
3. WHEN comparing chunks for deduplication THEN the system SHALL use SIMD-accelerated byte comparison where available.
4. WHEN computing rolling hashes THEN the system SHALL explore gear-hash SIMD implementations for FastCDC acceleration.
5. WHEN the target CPU supports AVX-512 THEN the system SHALL provide a feature flag to enable wider vector operations.

---

### Requirement 9: Memory Limits and Budgets

**User Story:** As an operations engineer, I want configurable memory limits, so that the chunker operates within container/VM memory constraints.

#### Acceptance Criteria

1. WHEN `MemoryBudget` is configured THEN the system SHALL track total allocations for buffers, caches, and scratch space.
2. WHEN memory usage approaches `soft_limit_bytes` THEN the system SHALL proactively evict cache entries and reduce buffer sizes.
3. WHEN memory usage exceeds `hard_limit_bytes` THEN the system SHALL return `Error::MemoryLimitExceeded` rather than allocate.
4. WHEN streaming large files THEN the system SHALL enforce `max_buffer_size` to prevent unbounded growth (current: 256MB ceiling).
5. WHEN batch processing THEN the system SHALL limit concurrent operations based on per-operation memory estimate.
6. IF memory tracking is enabled THEN the system SHALL report: `current_usage_bytes`, `peak_usage_bytes`, `allocation_count`.

---

### Requirement 10: CPU Quota Awareness

**User Story:** As a cluster administrator, I want the chunker to respect CPU quotas, so that it coexists with other workloads.

#### Acceptance Criteria

1. WHEN `CpuQuota` is configured THEN the system SHALL limit Rayon thread pool size to `max_threads`.
2. WHEN running in a cgroup-limited container THEN the system SHALL auto-detect available CPUs via `/sys/fs/cgroup` or `num_cpus` crate.
3. WHEN `nice_level` is configured THEN the system SHALL set thread priority to reduce impact on latency-sensitive workloads.
4. WHEN `cpu_time_limit_ms` is set for an operation THEN the system SHALL checkpoint progress and return partial results if exceeded.
5. WHEN cooperative scheduling is enabled THEN long-running chunking operations SHALL yield periodically (every `yield_interval_bytes`).

---

### Requirement 11: Graceful Resource Exhaustion Handling

**User Story:** As a reliability engineer, I want graceful degradation under resource pressure, so that the system remains stable.

#### Acceptance Criteria

1. WHEN memory allocation fails THEN the system SHALL attempt cache eviction and retry before returning an error.
2. WHEN file descriptor limits are approached THEN batch file processing SHALL throttle concurrent open files.
3. WHEN disk I/O latency spikes THEN the system SHALL implement exponential backoff for read retries.
4. WHEN decompression produces output exceeding `max_decompressed_size` THEN the system SHALL abort early (current: 1GB limit).
5. WHEN the Rayon thread pool is saturated THEN new tasks SHALL queue with bounded depth to prevent OOM.
6. IF graceful shutdown is requested THEN in-progress operations SHALL complete or checkpoint before termination.

---

### Requirement 12: Cooperative Scheduling

**User Story:** As a NIF developer, I want chunking to yield control periodically, so that the Erlang scheduler remains responsive.

#### Acceptance Criteria

1. WHEN running as a NIF THEN long-running operations SHALL call `enif_consume_timeslice` or yield after processing `yield_chunk_count` chunks.
2. WHEN streaming chunks via iterator THEN each `next()` call SHALL complete within `max_iteration_time_us` (default 1000us).
3. WHEN batch operations detect scheduler pressure THEN they SHALL reduce parallelism dynamically.
4. WHEN async chunking is used THEN the system SHALL use `tokio::task::yield_now()` between chunk emissions.
5. IF cooperative mode is disabled (standalone Rust use) THEN the system SHALL skip yield points for maximum throughput.

---

### Requirement 13: S3/Storage Backend Optimization

**User Story:** As a cloud architect, I want chunker output optimized for S3 storage, so that uploads are efficient and cost-effective.

#### Acceptance Criteria

1. WHEN emitting chunks THEN the system SHALL provide chunk size recommendations aligned with S3 multipart upload part sizes (5MB-5GB).
2. WHEN chunks are destined for S3 THEN the system SHALL support streaming upload without buffering entire chunks in memory.
3. WHEN chunk metadata is generated THEN the system SHALL include fields compatible with S3 object metadata: `Content-MD5`, `x-amz-meta-*`.
4. WHEN uploading to S3 THEN the system SHALL support concurrent part uploads with configurable `max_concurrent_uploads`.
5. WHEN S3 returns 503 SlowDown THEN the system SHALL implement exponential backoff with jitter.
6. WHEN chunks are stored THEN the system SHALL support content-addressable naming: `{hash_prefix}/{hash}` for efficient listing.

---

### Requirement 14: Network-Aware Chunking

**User Story:** As a distributed systems engineer, I want chunking parameters to adapt to network conditions, so that transfer efficiency is optimized.

#### Acceptance Criteria

1. WHEN `NetworkProfile::HighLatency` is selected THEN the system SHALL prefer larger chunks to reduce round-trips.
2. WHEN `NetworkProfile::LowBandwidth` is selected THEN the system SHALL prefer smaller chunks with aggressive compression.
3. WHEN bandwidth estimation is enabled THEN the system SHALL measure upload throughput and adjust chunk sizes dynamically.
4. WHEN network errors occur THEN the system SHALL support resumable uploads from the last successfully transferred chunk.
5. WHEN transferring over metered connections THEN the system SHALL report `bytes_transferred` for cost tracking.

---

### Requirement 15: Compression Ratio vs Speed Tradeoffs

**User Story:** As a pipeline developer, I want configurable compression strategies, so that I can balance storage costs against processing time.

#### Acceptance Criteria

1. WHEN `CompressionStrategy::Fastest` is selected THEN the system SHALL use LZ4 (current behavior).
2. WHEN `CompressionStrategy::Balanced` is selected THEN the system SHALL use zstd level 3 (current behavior).
3. WHEN `CompressionStrategy::Smallest` is selected THEN the system SHALL use zstd level 19 or XZ.
4. WHEN `CompressionStrategy::Adaptive` is selected THEN the system SHALL sample content and choose algorithm based on compressibility.
5. WHEN compressing NAR files THEN the system SHALL detect already-compressed content (JPEG, PNG, ZIP) and skip recompression.
6. WHEN compression ratio falls below `min_ratio_threshold` (default 0.95) THEN the system SHALL store uncompressed to save CPU.
7. IF compression time exceeds `max_compression_time_ms` per chunk THEN the system SHALL fall back to faster algorithm.

---

### Requirement 16: Adaptive Strategies Based on Content

**User Story:** As a performance engineer, I want the chunker to adapt to content characteristics, so that processing is optimized per-file.

#### Acceptance Criteria

1. WHEN the first 4KB of content is analyzed THEN the system SHALL detect content type: binary, text, compressed, multimedia.
2. WHEN content is already compressed (zstd, gzip, xz magic bytes) THEN the system SHALL skip compression and emit raw chunks.
3. WHEN content is highly repetitive (>90% single byte) THEN the system SHALL use run-length encoding hints in metadata.
4. WHEN content is text THEN the system SHALL prefer chunk boundaries at line breaks when within tolerance.
5. WHEN processing source code THEN the system SHALL optionally prefer boundaries at function/class definitions.
6. WHEN content entropy is measured THEN the system SHALL report `entropy_estimate` in chunk metadata for storage tier routing.

---

### Requirement 17: Performance Counters

**User Story:** As an SRE, I want real-time performance counters, so that I can monitor chunker health in production.

#### Acceptance Criteria

1. WHEN the `metrics` feature is enabled THEN the system SHALL emit counters via the `metrics` crate facade.
2. WHEN chunks are emitted THEN the system SHALL increment: `chunker.chunks_emitted_total`, `chunker.bytes_processed_total`.
3. WHEN compression completes THEN the system SHALL record: `chunker.compression_bytes_in`, `chunker.compression_bytes_out`, `chunker.compression_ratio`.
4. WHEN hashing completes THEN the system SHALL record: `chunker.hash_operations_total`, `chunker.hash_bytes_total`.
5. WHEN cache operations occur THEN the system SHALL record: `chunker.cache_hits_total`, `chunker.cache_misses_total`.
6. WHEN errors occur THEN the system SHALL increment: `chunker.errors_total{type="io|bounds|compression"}`.

---

### Requirement 18: Latency Histograms

**User Story:** As a performance engineer, I want latency distribution data, so that I can identify tail latency issues.

#### Acceptance Criteria

1. WHEN chunk processing completes THEN the system SHALL record latency in: `chunker.chunk_latency_seconds` histogram.
2. WHEN compression completes THEN the system SHALL record: `chunker.compression_latency_seconds` histogram by algorithm.
3. WHEN hashing completes THEN the system SHALL record: `chunker.hash_latency_seconds` histogram by algorithm.
4. WHEN batch operations complete THEN the system SHALL record: `chunker.batch_latency_seconds` histogram.
5. WHEN histograms are configured THEN the system SHALL support custom bucket boundaries for application-specific percentiles.
6. IF p99 latency exceeds `latency_alert_threshold_ms` THEN the system SHALL emit a warning log.

---

### Requirement 19: Throughput Tracking

**User Story:** As an operations engineer, I want throughput metrics, so that I can capacity plan and detect regressions.

#### Acceptance Criteria

1. WHEN processing data THEN the system SHALL maintain rolling `bytes_per_second` gauge updated every second.
2. WHEN batch processing THEN the system SHALL report: `files_per_second`, `chunks_per_second`.
3. WHEN throughput drops below `min_throughput_bytes_per_second` for >10 seconds THEN the system SHALL log a warning.
4. WHEN benchmarking THEN the system SHALL report: `peak_throughput`, `average_throughput`, `throughput_std_dev`.
5. WHEN throughput metrics are queried THEN the system SHALL return sliding window statistics (1m, 5m, 15m).

---

### Requirement 20: Bottleneck Identification Tools

**User Story:** As a performance engineer, I want built-in profiling tools, so that I can identify and fix performance bottlenecks.

#### Acceptance Criteria

1. WHEN `CHUNKER_PROFILE=1` environment variable is set THEN the system SHALL emit detailed timing spans via `tracing`.
2. WHEN profiling is enabled THEN the system SHALL report time spent in: FastCDC, hashing, compression, I/O wait, serialization.
3. WHEN the `flamegraph` feature is enabled THEN the system SHALL support integration with `tracing-flame` for visualization.
4. WHEN bottleneck analysis is requested THEN the system SHALL identify the slowest phase as percentage of total time.
5. WHEN memory profiling is enabled THEN the system SHALL track allocation hotspots via `tracing-allocator`.
6. IF I/O wait exceeds 50% of total time THEN the system SHALL suggest increasing read buffer sizes.

---

### Requirement 21: Extensible Algorithm Selection

**User Story:** As a library maintainer, I want pluggable algorithm implementations, so that new algorithms can be added without API breaks.

#### Acceptance Criteria

1. WHEN selecting a chunking algorithm THEN the system SHALL support: `ChunkingAlgorithm::FastCDC`, `ChunkingAlgorithm::RabinKarp`, `ChunkingAlgorithm::Fixed`.
2. WHEN selecting a hash algorithm THEN the system SHALL support registration of custom algorithms via `HashAlgorithm::Custom(name, fn)`.
3. WHEN selecting a compression algorithm THEN the system SHALL support runtime codec registration.
4. WHEN algorithm parameters are configured THEN the system SHALL validate via the algorithm's `validate_params()` method.
5. WHEN a new algorithm is registered THEN it SHALL automatically integrate with caching, metrics, and tracing.
6. IF an algorithm is deprecated THEN the system SHALL emit a compile-time warning and runtime log.

---

### Requirement 22: New Compression Codec Support

**User Story:** As a storage architect, I want to easily add new compression codecs, so that I can adopt emerging standards.

#### Acceptance Criteria

1. WHEN implementing a new codec THEN the developer SHALL implement the `CompressionCodec` trait with: `compress`, `decompress`, `magic_bytes`, `name`.
2. WHEN auto-detection is used THEN the system SHALL match magic bytes against all registered codecs.
3. WHEN a codec is registered THEN it SHALL automatically appear in `CompressionStrategy::Auto` selection.
4. WHEN codec benchmarks are run THEN results SHALL be comparable across all registered codecs.
5. WHEN codec support is optional THEN it SHALL be gated behind feature flags to minimize binary size.
6. IF a codec is not available at runtime (missing native lib) THEN the system SHALL return `Error::CodecUnavailable`.

---

### Requirement 23: Hardware Acceleration Readiness

**User Story:** As a performance engineer, I want the library prepared for hardware acceleration, so that we can leverage GPUs/FPGAs when available.

#### Acceptance Criteria

1. WHEN the `hw-accel` feature is enabled THEN the system SHALL detect available acceleration: Intel QAT, NVIDIA nvCOMP, AWS Graviton.
2. WHEN hardware compression is available THEN the system SHALL transparently offload eligible operations.
3. WHEN hardware is unavailable THEN the system SHALL fall back to software implementations without error.
4. WHEN hardware operations complete THEN the system SHALL verify output matches software implementation (configurable validation).
5. WHEN multiple accelerators are available THEN the system SHALL support selection via `HardwarePreference` config.
6. IF hardware acceleration fails THEN the system SHALL log the failure and continue with software fallback.
7. WHEN profiling THEN the system SHALL report: `hw_accel_operations`, `hw_accel_bytes`, `hw_accel_speedup_ratio`.

---

### Requirement 24: Async Runtime Compatibility

**User Story:** As an application developer, I want the chunker to work with my async runtime, so that I can integrate it into async pipelines.

#### Acceptance Criteria

1. WHEN the `async-stream` feature is enabled THEN the system SHALL provide `futures::Stream` implementations.
2. WHEN async operations are used THEN the system SHALL be runtime-agnostic (no hardcoded Tokio dependency in library code).
3. WHEN blocking operations are necessary THEN the system SHALL document spawn_blocking requirements.
4. WHEN streaming chunks asynchronously THEN memory usage SHALL remain bounded regardless of consumer backpressure.
5. WHEN async cancellation occurs THEN resources SHALL be properly cleaned up (no leaked buffers or file handles).

---

### Requirement 25: Thread Safety and Concurrent Access

**User Story:** As a systems programmer, I want clear thread-safety guarantees, so that I can safely share chunker instances.

#### Acceptance Criteria

1. WHEN `ChunkingOptions` is used THEN it SHALL be `Send + Sync` for safe sharing across threads.
2. WHEN `CompressionScratch` is used THEN it SHALL NOT be `Sync` (requires exclusive access) but SHALL be `Send`.
3. WHEN caches are enabled THEN they SHALL use `Arc<DashMap>` or similar for concurrent access without external locking.
4. WHEN worker pools are spawned THEN the system SHALL properly propagate panics to the caller.
5. IF a data race is detected during testing THEN the CI pipeline SHALL fail via MIRI or ThreadSanitizer.

---

## Non-Functional Requirements

### NFR-1: Performance Benchmarks

1. WHEN benchmarked on reference hardware (AMD EPYC 7763, 256GB RAM) THEN chunking throughput SHALL exceed 2 GB/s for 10MB files.
2. WHEN benchmarked THEN SHA256 throughput SHALL exceed 1 GB/s (with SHA-NI).
3. WHEN benchmarked THEN BLAKE3 throughput SHALL exceed 5 GB/s (with SIMD).
4. WHEN benchmarked THEN zstd level 3 compression SHALL exceed 500 MB/s.
5. WHEN benchmarked THEN LZ4 compression SHALL exceed 2 GB/s.
6. WHEN benchmarked THEN p99 latency for 1MB chunk SHALL be under 5ms.

### NFR-2: Resource Efficiency

1. WHEN processing 100MB files THEN peak memory usage SHALL not exceed 50MB (excluding input buffer).
2. WHEN idle THEN background threads SHALL consume <1% CPU.
3. WHEN using caches THEN memory overhead per cached entry SHALL not exceed 1KB.
4. WHEN NIF operations complete THEN all Rust-allocated memory SHALL be freed or transferred to BEAM.

### NFR-3: Reliability

1. WHEN fuzz testing THEN the system SHALL handle arbitrary input without panics for 24+ hours.
2. WHEN processing malformed data THEN the system SHALL return errors, not crash.
3. WHEN decompression bombs are detected THEN the system SHALL abort within 100ms.
4. WHEN integration tests run THEN all code paths SHALL achieve >90% coverage.

### NFR-4: Observability

1. WHEN tracing is enabled THEN all public functions SHALL have `#[instrument]` spans.
2. WHEN errors occur THEN structured logs SHALL include: operation, input_size, error_type, duration.
3. WHEN metrics are exported THEN they SHALL be compatible with Prometheus, StatsD, and OpenTelemetry.

### NFR-5: Compatibility

1. WHEN compiled THEN the library SHALL support Rust 2024 edition (current).
2. WHEN used as NIF THEN the library SHALL support Erlang/OTP 24+ and Elixir 1.14+.
3. WHEN cross-compiled THEN the library SHALL support: x86_64-linux, aarch64-linux, x86_64-darwin, aarch64-darwin.
4. WHEN ABI stability is required THEN public structs SHALL use `#[repr(C)]` where appropriate.

---

## Glossary

- **NAR**: Nix Archive format, a deterministic archive format used by Nix
- **FastCDC**: Fast Content-Defined Chunking algorithm for deduplication
- **Gear hash**: Rolling hash algorithm used by FastCDC
- **NIF**: Native Implemented Function, Erlang/Elixir FFI mechanism
- **SIMD**: Single Instruction Multiple Data, CPU vector extensions
- **SHA-NI**: Intel SHA hardware acceleration instructions
- **QAT**: Intel QuickAssist Technology for hardware compression
