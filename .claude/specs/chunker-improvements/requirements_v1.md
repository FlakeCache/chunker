# Requirements Document: World-Class Performance for Chunker Library

## Introduction

This document defines comprehensive requirements for optimizing the Chunker library to achieve world-class performance suitable for processing hundreds of millions of chunking operations. The library is a high-performance content-defined chunking system for Nix NARs, written in Rust 2024 edition, currently at version 0.1.0-beta6.

The optimization effort targets multiple layers: algorithmic improvements, memory management, SIMD acceleration, parallel processing tuning, and comprehensive benchmarking infrastructure to detect regressions and guide optimization decisions.

**Target Performance Characteristics:**
- Sub-millisecond latency for typical chunk operations
- Linear scalability with data size
- Minimal memory allocation overhead in hot paths
- Effective parallelization for multi-core systems
- Zero-copy data paths where feasible

---

## Requirements

### Requirement 1: Comprehensive Benchmark Suite

**User Story:** As a library maintainer, I want a comprehensive benchmark suite that covers all critical code paths, so that I can measure performance accurately and identify optimization opportunities.

#### Acceptance Criteria

1. WHEN benchmarks are executed THEN the system SHALL measure throughput (bytes/second) for chunking operations across multiple data sizes (1KB, 64KB, 1MB, 10MB, 100MB, 1GB).

2. WHEN benchmarks are executed THEN the system SHALL measure latency distributions (p50, p95, p99, p999) for individual chunk operations.

3. WHEN benchmarks are executed THEN the system SHALL compare SHA256 vs BLAKE3 hashing performance with identical data sets.

4. WHEN benchmarks are executed THEN the system SHALL measure FastCDC boundary detection throughput separately from hash computation.

5. WHEN benchmarks are executed THEN the system SHALL test both streaming (`ChunkStream`) and eager (`chunk_data`) APIs with identical inputs.

6. WHEN benchmarks are executed THEN the system SHALL measure compression codec performance (zstd, lz4, xz, bzip2) at multiple compression levels.

7. WHEN benchmarks are executed THEN the system SHALL use realistic data patterns including: all-zeros, random bytes, repeating patterns, and actual NAR file samples.

8. WHEN benchmarks are executed THEN the system SHALL measure memory allocation counts and peak memory usage per operation.

9. WHEN benchmarks are executed THEN the system SHALL output results in machine-parseable format (JSON) for CI integration.

---

### Requirement 2: Performance Regression Detection

**User Story:** As a library maintainer, I want automated regression detection in CI, so that performance degradations are caught before merging.

#### Acceptance Criteria

1. WHEN a pull request is submitted THEN the CI pipeline SHALL run the benchmark suite against the main branch baseline.

2. WHEN benchmark results deviate by more than 5% from baseline THEN the system SHALL flag the change as a potential regression.

3. WHEN a regression is detected THEN the system SHALL report which specific benchmark(s) regressed and by what percentage.

4. WHEN benchmarks complete THEN the system SHALL store historical results for trend analysis.

5. WHEN running regression detection THEN the system SHALL use statistical significance testing (multiple iterations, confidence intervals) to reduce false positives.

6. IF benchmark variance exceeds 10% across runs THEN the system SHALL report the benchmark as unstable and exclude it from regression gates.

---

### Requirement 3: Profiling Infrastructure

**User Story:** As a developer, I want integrated profiling tools, so that I can identify hot spots and optimize effectively.

#### Acceptance Criteria

1. WHEN profiling mode is enabled THEN the system SHALL support CPU profiling via `perf` or `flamegraph` integration.

2. WHEN profiling mode is enabled THEN the system SHALL support heap profiling to identify allocation hot spots.

3. WHEN profiling mode is enabled THEN the system SHALL support cache miss analysis via `perf stat` or `cachegrind`.

4. WHEN profiling is requested THEN the system SHALL generate flamegraphs as SVG artifacts.

5. WHEN profiling chunking operations THEN the system SHALL separately report time spent in: boundary detection, hashing, memory allocation, and buffer management.

---

### Requirement 4: Parallel Threshold Optimization

**User Story:** As a user processing large files, I want optimal parallelization, so that multi-core systems are fully utilized without overhead for small workloads.

#### Acceptance Criteria

1. WHEN fewer than N chunks are detected (configurable threshold) THEN the system SHALL use sequential processing to avoid Rayon overhead.

2. WHEN the parallel threshold is evaluated THEN the system SHALL consider both chunk count AND total data size.

3. WHEN parallelization is used THEN the system SHALL achieve at least 80% CPU utilization across available cores for large workloads.

4. WHEN the current parallel threshold of 4+ chunks is evaluated THEN the system SHALL benchmark to determine optimal threshold based on typical chunk sizes (256KB-4MB).

5. IF environment variable `CHUNKER_PARALLEL_THRESHOLD` is set THEN the system SHALL use that value to override the default threshold.

6. WHEN operating on systems with different core counts THEN the system SHALL adapt parallelization strategy accordingly.

---

### Requirement 5: Metrics Overhead Reduction

**User Story:** As a user running hundreds of millions of operations, I want minimal metrics overhead, so that observability does not impact throughput.

#### Acceptance Criteria

1. WHEN metrics are recorded per chunk THEN the overhead SHALL be less than 100 nanoseconds per chunk.

2. WHEN the current 3-metrics-per-chunk pattern (counter, counter, histogram) is evaluated THEN the system SHALL provide option for batched metric updates.

3. IF metrics feature is disabled at compile time THEN the system SHALL compile out all metrics code with zero runtime cost.

4. WHEN metrics are enabled THEN the system SHALL use thread-local aggregation to avoid atomic contention.

5. WHEN histogram metrics are recorded THEN the system SHALL use pre-computed bucket boundaries to minimize computation.

6. IF environment variable `CHUNKER_METRICS_SAMPLE_RATE` is set THEN the system SHALL sample metrics at the specified rate (e.g., 1 in 1000) to reduce overhead.

---

### Requirement 6: Memory Allocation Elimination in Hot Paths

**User Story:** As a user processing high volumes, I want zero allocations in the chunking hot path, so that GC/allocator pressure is minimized.

#### Acceptance Criteria

1. WHEN processing chunks in `chunk_data_with_hash` THEN the hashing pass SHALL NOT allocate memory beyond the initial results vector.

2. WHEN using `ChunkStream` iterator THEN the system SHALL reuse internal buffers across iterations where possible.

3. WHEN returning `ChunkMetadata` THEN the payload field SHALL use zero-copy `Bytes` slices from the source buffer.

4. WHEN processing batch operations THEN the system SHALL pre-allocate result vectors based on estimated chunk count.

5. WHEN using SHA256 hasher THEN the system SHALL reuse hasher instances via object pooling or thread-local storage.

6. IF arena allocator is available THEN the system SHALL support arena-based allocation for temporary chunk processing buffers.

---

### Requirement 7: Buffer Pooling for Compression Operations

**User Story:** As a user compressing many chunks, I want buffer reuse, so that allocation overhead is minimized.

#### Acceptance Criteria

1. WHEN `CompressionScratch` is used THEN the system SHALL maintain buffer capacity across multiple compress/decompress operations.

2. WHEN thread-local buffer pools are requested THEN the system SHALL provide a `ThreadLocal<CompressionScratch>` pattern for zero-contention reuse.

3. WHEN buffer pools are used THEN the system SHALL limit maximum pool size to prevent unbounded memory growth.

4. WHEN compression operations complete THEN the system SHALL NOT shrink buffers below a configurable minimum capacity.

5. WHEN multiple compression strategies are used THEN the system SHALL share underlying buffers where data layout permits.

---

### Requirement 8: BLAKE3 Performance Optimization

**User Story:** As a user choosing hash algorithms, I want BLAKE3 to deliver maximum performance, so that I can benefit from its speed advantages.

#### Acceptance Criteria

1. WHEN BLAKE3 is used THEN the system SHALL enable SIMD acceleration (AVX2/AVX-512/NEON) based on runtime CPU detection.

2. WHEN BLAKE3 is used for parallel chunk hashing THEN the system SHALL leverage BLAKE3's native parallelism for large chunks.

3. WHEN comparing SHA256 vs BLAKE3 THEN benchmarks SHALL demonstrate BLAKE3 throughput of at least 3x SHA256 on modern CPUs.

4. WHEN building with `--features asm` THEN the system SHALL use assembly-optimized implementations for both SHA256 and BLAKE3.

5. WHEN BLAKE3 Rayon feature is enabled THEN the system SHALL avoid double-parallelization with Rayon chunk processing.

---

### Requirement 9: FastCDC Optimization

**User Story:** As a user chunking large files, I want FastCDC to operate at maximum efficiency, so that boundary detection is not the bottleneck.

#### Acceptance Criteria

1. WHEN FastCDC processes data THEN the gear table lookup SHALL be cache-friendly (sequential memory access patterns).

2. WHEN chunk sizes are specified THEN the system SHALL validate that sizes fit in u32 without overflow (current bug at lines 214-216).

3. WHEN average chunk size is tuned THEN the system SHALL provide guidance on optimal sizes for NAR patterns (typically 256KB-1MB).

4. WHEN FastCDC is initialized THEN the gear table SHALL be computed once and cached statically.

5. IF custom gear tables are supported THEN the system SHALL allow NAR-optimized gear tables that improve boundary distribution.

---

### Requirement 10: SIMD Base32 Encoding

**User Story:** As a user encoding many hashes to Nix base32, I want SIMD-accelerated encoding, so that this common operation is fast.

#### Acceptance Criteria

1. WHEN encoding to Nix base32 THEN the system SHALL support SIMD-accelerated encoding on x86_64 (AVX2) and aarch64 (NEON).

2. WHEN SIMD is not available THEN the system SHALL fall back to the current scalar implementation transparently.

3. WHEN SIMD base32 is used THEN throughput SHALL be at least 2x the scalar implementation for 32-byte inputs.

4. WHEN decoding from Nix base32 THEN the system SHALL use lookup table optimization with the existing `NIX_BASE32_INVERSE` table.

5. WHEN encoding batch operations (multiple hashes) THEN the system SHALL process multiple hashes in parallel using SIMD.

---

### Requirement 11: Zero-Copy Data Paths

**User Story:** As a user processing large files, I want zero-copy data handling, so that memory bandwidth is maximized.

#### Acceptance Criteria

1. WHEN `ChunkMetadata` payload is created THEN the system SHALL use `Bytes::slice` for zero-copy views into the source buffer.

2. WHEN chunks are passed to compression THEN the system SHALL accept borrowed slices without requiring owned copies.

3. WHEN NIF bindings receive Elixir binaries THEN the system SHALL process them without copying to Rust-owned memory where possible.

4. WHEN streaming from `AsyncRead` sources THEN the system SHALL minimize intermediate buffer copies.

5. WHEN returning results to callers THEN the system SHALL provide options for borrowed vs owned data to avoid unnecessary copies.

---

### Requirement 12: Cache-Friendly Data Structures

**User Story:** As a performance-critical application, I want cache-optimized data layouts, so that CPU cache efficiency is maximized.

#### Acceptance Criteria

1. WHEN `ChunkMetadata` is stored in vectors THEN the layout SHALL be optimized for sequential access patterns.

2. WHEN processing chunks in parallel THEN the system SHALL avoid false sharing by padding or alignment.

3. WHEN iterating over chunk results THEN hot fields (hash, offset, length) SHALL be contiguous in memory.

4. IF Structure-of-Arrays (SoA) layout provides benefit THEN the system SHALL offer SoA alternatives to current Array-of-Structures.

5. WHEN buffer sizes are chosen THEN the system SHALL align to cache line boundaries (64 bytes) for optimal performance.

---

### Requirement 13: Compression Dictionary Preloading

**User Story:** As a user compressing NAR data, I want preloaded compression dictionaries, so that compression ratios and speed are optimized for NAR patterns.

#### Acceptance Criteria

1. WHEN zstd compression is used THEN the system SHALL support preloaded `EncoderDictionary` for NAR-optimized compression.

2. WHEN dictionaries are used THEN they SHALL be trained on representative NAR samples for optimal effectiveness.

3. WHEN dictionary compression is enabled THEN the system SHALL achieve at least 10% better compression ratio than default zstd level 3.

4. WHEN decompressing with dictionaries THEN the system SHALL support preloaded `DecoderDictionary` to avoid repeated parsing.

5. IF dictionary is not available THEN the system SHALL fall back to standard compression transparently.

---

### Requirement 14: Compression Level Tuning

**User Story:** As a user balancing speed and compression, I want tuned compression levels, so that I get optimal trade-offs for my use case.

#### Acceptance Criteria

1. WHEN using `CompressionStrategy::Balanced` THEN the system SHALL use zstd level 3 as the default (current behavior validated).

2. WHEN using `CompressionStrategy::Fastest` THEN the system SHALL achieve at least 2GB/s compression throughput on modern CPUs.

3. WHEN compression level is specified THEN the system SHALL provide benchmark data showing speed/ratio trade-offs for levels 1-19.

4. IF adaptive compression is enabled THEN the system SHALL adjust compression level based on data compressibility detected in early chunks.

5. WHEN parallel compression is used THEN the system SHALL distribute chunks across compression worker threads efficiently.

---

### Requirement 15: Parallel Compression Strategies

**User Story:** As a user compressing many chunks, I want parallel compression, so that multi-core systems are fully utilized.

#### Acceptance Criteria

1. WHEN compressing multiple chunks THEN the system SHALL support parallel compression via worker pools (extending current `spawn_zstd_worker`).

2. WHEN parallel compression is used THEN the system SHALL maintain chunk ordering in results.

3. WHEN worker count is determined THEN the system SHALL default to `num_cpus - 1` to leave headroom for other operations.

4. WHEN worker pools are used THEN they SHALL support backpressure to prevent unbounded memory growth.

5. IF compression workers are idle THEN the system SHALL support work stealing from the hashing pipeline.

---

### Requirement 16: Bzip2 Decompression Bomb Protection

**User Story:** As a security-conscious user, I want protection against decompression bombs, so that malicious inputs cannot exhaust memory.

#### Acceptance Criteria

1. WHEN decompressing bzip2 data THEN the system SHALL enforce the `MAX_DECOMPRESSED_SIZE` limit (current bug at line 714 uses `==` instead of `>=`).

2. WHEN decompressed size equals the limit THEN the system SHALL treat this as `SizeExceeded` to prevent boundary condition exploits.

3. WHEN any decompression exceeds the limit THEN the system SHALL return `CompressionError::SizeExceeded` immediately.

4. WHEN decompression bombs are detected THEN the system SHALL log a warning via `tracing::warn`.

5. WHEN configurable limits are supported THEN the system SHALL allow per-operation size limits via function parameters.

---

### Requirement 17: Integer Overflow Protection

**User Story:** As a user handling large files, I want integer overflow protection, so that undefined behavior and incorrect results are prevented.

#### Acceptance Criteria

1. WHEN chunk sizes are converted to u32 for FastCDC THEN the system SHALL use checked conversions with explicit error handling (current issue at lines 214-216).

2. WHEN offset + length is computed THEN the system SHALL use `checked_add` to prevent overflow.

3. WHEN position counters are updated THEN the system SHALL handle u64 overflow for files larger than 16 exabytes gracefully.

4. IF integer overflow is detected THEN the system SHALL return `ChunkingError::Bounds` or a new `ChunkingError::Overflow` variant.

5. WHEN casting between integer types THEN the system SHALL use safe casting macros or functions with explicit bounds checking.

---

### Requirement 18: Branch Prediction Optimization

**User Story:** As a performance-critical application, I want optimized branch prediction, so that CPU pipeline stalls are minimized.

#### Acceptance Criteria

1. WHEN common code paths are identified THEN the system SHALL use `#[cold]` attributes for error handling paths.

2. WHEN hot loops are identified THEN the system SHALL use `std::hint::likely`/`unlikely` for predictable branches.

3. WHEN match statements are used in hot paths THEN the system SHALL order arms by frequency of occurrence.

4. WHEN bounds checks are performed THEN the system SHALL structure code to enable compiler optimization of redundant checks.

5. WHEN profiling data is available THEN the system SHALL use PGO (Profile-Guided Optimization) in release builds.

---

### Requirement 19: CPU Feature Detection and Dispatch

**User Story:** As a library supporting multiple platforms, I want runtime CPU feature detection, so that optimal code paths are selected automatically.

#### Acceptance Criteria

1. WHEN SIMD operations are available THEN the system SHALL detect AVX2, AVX-512, and NEON at runtime.

2. WHEN CPU features are detected THEN the system SHALL cache the detection result for the process lifetime.

3. WHEN multiple implementations exist THEN the system SHALL dispatch to the fastest available implementation transparently.

4. WHEN cross-compiling THEN the system SHALL support `target_feature` for static feature selection.

5. WHEN feature detection fails THEN the system SHALL fall back to portable scalar implementations.

---

### Requirement 20: Async Stream Optimization

**User Story:** As a user with async workloads, I want optimized async streaming, so that chunking integrates efficiently with async runtimes.

#### Acceptance Criteria

1. WHEN using `ChunkStreamAsync` THEN the system SHALL avoid blocking the async runtime with CPU-bound work.

2. WHEN async buffers are sized THEN the system SHALL use adaptive sizing based on observed throughput.

3. WHEN `effective_async_buffer_limit()` is called THEN the system SHALL respect environment variable bounds efficiently.

4. WHEN async operations yield THEN the system SHALL yield at chunk boundaries to enable cooperative scheduling.

5. IF `spawn_blocking` is available THEN the system SHALL offload CPU-intensive hashing to blocking thread pools.

---

### Requirement 21: Tracing and Logging Optimization

**User Story:** As a user with high-volume workloads, I want minimal tracing overhead, so that observability does not impact performance.

#### Acceptance Criteria

1. WHEN tracing is disabled at compile time THEN the system SHALL compile out all tracing code with zero cost.

2. WHEN tracing is enabled THEN the system SHALL use sampling (current `TRACE_SAMPLE_EVERY = 1024`) to reduce overhead.

3. WHEN `tracing::instrument` is used THEN hot paths SHALL use `skip_all` to minimize argument capture overhead.

4. WHEN trace levels are checked THEN the system SHALL use `tracing::enabled!` macro for efficient level checking.

5. IF structured logging is used THEN field serialization SHALL be deferred until log emission is confirmed.

---

### Requirement 22: NIF Binding Optimization

**User Story:** As an Elixir user, I want optimized NIF bindings, so that cross-language overhead is minimized.

#### Acceptance Criteria

1. WHEN NIF functions are called THEN the system SHALL minimize Erlang/Rust boundary crossings per operation.

2. WHEN binary data is passed to NIFs THEN the system SHALL use `Binary` references without copying where possible.

3. WHEN returning results to Elixir THEN the system SHALL use `OwnedBinary` efficiently with pre-allocated capacity.

4. WHEN long-running operations are performed THEN the system SHALL use `schedule = "DirtyCpu"` (current behavior confirmed).

5. IF batch operations are requested THEN the system SHALL process batches in single NIF calls to amortize crossing overhead.

---

### Requirement 23: Documentation and Optimization Guides

**User Story:** As a library user, I want performance documentation, so that I can configure the library optimally for my use case.

#### Acceptance Criteria

1. WHEN performance documentation is provided THEN it SHALL include benchmark results for common configurations.

2. WHEN tuning guidance is provided THEN it SHALL cover: chunk sizes, parallel thresholds, buffer sizes, and hash algorithm selection.

3. WHEN environment variables are documented THEN all performance-related variables SHALL be listed with their effects.

4. WHEN optimization is documented THEN it SHALL include case studies for NAR-specific workloads.

5. WHEN API documentation is provided THEN it SHALL include performance characteristics for each public function.

---

### Requirement 24: Continuous Performance Monitoring

**User Story:** As a library maintainer, I want continuous performance monitoring, so that long-term performance trends are visible.

#### Acceptance Criteria

1. WHEN benchmarks are run in CI THEN results SHALL be stored in a time-series format for historical analysis.

2. WHEN performance dashboards are provided THEN they SHALL show throughput, latency, and memory usage trends over time.

3. WHEN releases are tagged THEN benchmark results SHALL be associated with the release version.

4. IF performance improves significantly THEN the change SHALL be highlighted in release notes.

5. WHEN comparing across versions THEN the system SHALL provide diff reports showing performance changes.

---

## Non-Functional Requirements

### NFR-1: Backward Compatibility

1. WHEN optimizations are applied THEN existing public API signatures SHALL remain unchanged.
2. WHEN new APIs are introduced THEN they SHALL be additive and not break existing code.
3. WHEN default behaviors change THEN they SHALL be configurable to restore previous behavior.

### NFR-2: Build Time Impact

1. WHEN optimization features are added THEN incremental build time SHALL increase by no more than 20%.
2. WHEN SIMD code is added THEN it SHALL be feature-gated to allow opting out.
3. WHEN benchmarks are added THEN they SHALL be in separate compilation units from library code.

### NFR-3: Binary Size

1. WHEN optimizations are added THEN release binary size SHALL increase by no more than 10%.
2. WHEN multiple implementations exist THEN unused implementations SHALL be dead-code eliminated.
3. WHEN SIMD code is included THEN only detected-at-runtime implementations SHALL be linked.

### NFR-4: Platform Support

1. WHEN SIMD optimizations are added THEN x86_64 (AVX2) and aarch64 (NEON) SHALL be supported.
2. WHEN platform-specific code is added THEN fallback implementations SHALL exist for other platforms.
3. WHEN cross-compilation is used THEN all optimizations SHALL work correctly.

---

## Appendix: Current Implementation Analysis

### Identified Issues

1. **Bzip2 Decompression Bomb Bug (compression.rs:714):** The check `if (output.len() - start_len) as u64 == MAX_DECOMPRESSED_SIZE` should use `>=` to properly detect size exceeded conditions.

2. **Integer Overflow Risk (chunking.rs:214-216):** The casts `options.min_size as u32`, `options.avg_size as u32`, `options.max_size as u32` can overflow if sizes exceed u32::MAX.

3. **Metrics Overhead:** Current implementation emits 3 metrics per chunk (lines 484-486 in streaming path), which may be excessive for high-volume workloads.

4. **Parallel Threshold:** Current threshold of 4+ chunks (line 515) may not be optimal for typical chunk sizes of 256KB-4MB.

### Performance Baseline

Current benchmarks measure:
- FastCDC raw throughput on 10MB data
- ChunkStream streaming throughput on 10MB data
- Eager chunk_data throughput on 10MB data
- SHA256 and BLAKE3 hashing on 1MB data
- Zstd and LZ4 compression on 1MB data

### Recommended Priority Order

1. Fix security issues (bzip2 bomb, overflow protection) - Requirement 16, 17
2. Establish comprehensive benchmarking - Requirement 1, 2, 3
3. Optimize hot paths (metrics, allocation) - Requirement 5, 6
4. BLAKE3 optimization - Requirement 8
5. Parallel threshold tuning - Requirement 4
6. SIMD acceleration - Requirement 10, 19
7. Compression optimization - Requirement 13, 14, 15
8. Zero-copy and cache optimization - Requirement 11, 12
