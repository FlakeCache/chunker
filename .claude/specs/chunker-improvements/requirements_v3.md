# Requirements Document: World-Class Performance for Chunker Library

## Introduction

This document defines requirements for transforming the Chunker library into a world-class, high-performance content-defined chunking solution capable of handling hundreds of millions of chunking operations with sub-millisecond latency. The Chunker library is a Rust-based system providing FastCDC chunking, multi-codec compression (zstd, xz, lz4, bzip2), cryptographic hashing (SHA256, BLAKE3), and Ed25519 signing, with Elixir NIF bindings for integration with the FlakeCache ecosystem.

The library currently demonstrates solid architecture with parallel hashing via Rayon, streaming chunkers, and scratch buffer reuse patterns. However, to achieve world-class performance for 100s of millions of operations, critical bugs must be fixed, comprehensive benchmarking infrastructure must be established, and platform-specific optimizations must be implemented.

## Requirements

### Requirement 1: Bzip2 Decompression Bomb Fix

**User Story:** As a system operator, I want the bzip2 decompression to correctly detect decompression bombs, so that malicious inputs cannot cause memory exhaustion.

#### Acceptance Criteria

1. WHEN the bzip2 decompressor reads exactly `MAX_DECOMPRESSED_SIZE` bytes THEN the system SHALL continue reading to detect if additional data exists beyond the limit (current code at line 714 uses `==` instead of `>` comparison, which fails to detect bombs that decompress to exactly the limit).

2. WHEN bzip2 decompression output exceeds `MAX_DECOMPRESSED_SIZE` THEN the system SHALL return `CompressionError::SizeExceeded`.

3. IF a legitimate file decompresses to exactly `MAX_DECOMPRESSED_SIZE` bytes THEN the system SHALL successfully decompress it without false-positive rejection.

4. WHEN implementing the fix THEN the system SHALL use the same `take(limit + 1)` pattern used in `decompress_zstd_into_with_limit` and `decompress_lz4_into` functions.

---

### Requirement 2: Integer Overflow Protection in FastCDC Size Casting

**User Story:** As a developer, I want chunking size parameters to be safely converted to u32, so that large size values do not silently truncate and cause undefined behavior.

#### Acceptance Criteria

1. WHEN `ChunkingOptions` values (min_size, avg_size, max_size) are cast from `usize` to `u32` for FastCDC initialization (lines 214-216 in chunking.rs) THEN the system SHALL validate that values fit within u32 range before casting.

2. IF any chunking size parameter exceeds `u32::MAX` THEN the system SHALL return `ChunkingError::InvalidOptions` with a descriptive message.

3. WHEN the `ChunkingOptions::validate()` method runs THEN it SHALL include overflow checks for u32 compatibility in addition to existing logical constraints.

4. WHILE processing chunk definitions from FastCDC THEN the system SHALL use checked arithmetic for offset + length calculations to prevent overflow.

---

### Requirement 3: Atomic Ordering Correctness for TRACE_SAMPLE_COUNTER

**User Story:** As a performance engineer, I want atomic operations to use appropriate memory ordering, so that trace sampling is correct and performant across all platforms.

#### Acceptance Criteria

1. WHEN `TRACE_SAMPLE_COUNTER.fetch_add(1, Ordering::Relaxed)` is called THEN the system SHALL continue using `Relaxed` ordering since the counter is only used for statistical sampling and does not synchronize other memory operations.

2. WHERE the atomic counter is used for trace sampling THEN the implementation SHALL document why `Relaxed` ordering is sufficient (no happens-before relationship required).

3. IF future functionality requires synchronization based on the counter value THEN the system SHALL upgrade to `Ordering::AcqRel` or stronger.

---

### Requirement 4: Micro-Benchmark Suite for All Core Functions

**User Story:** As a performance engineer, I want comprehensive micro-benchmarks for every core function, so that I can identify performance bottlenecks and track improvements.

#### Acceptance Criteria

1. WHEN running benchmarks THEN the system SHALL measure throughput (bytes/second) for:
   - FastCDC raw chunking (without hashing)
   - `chunk_data` (eager, in-memory)
   - `ChunkStream` (streaming)
   - SHA256 hashing
   - BLAKE3 hashing
   - zstd compression (levels 1, 3, 9, 19)
   - lz4 compression
   - xz compression
   - bzip2 compression
   - Ed25519 signing
   - Ed25519 verification
   - Nix base32 encode/decode

2. WHEN running benchmarks THEN the system SHALL test with multiple data sizes: 1KB, 64KB, 1MB, 10MB, 100MB, 1GB.

3. WHEN running benchmarks THEN the system SHALL test with multiple data patterns: zeros, random, realistic NAR content (mixed binary/text).

4. WHEN benchmarks complete THEN the system SHALL output results in machine-parseable format (JSON) for CI integration.

5. WHEN benchmarks run in CI THEN the system SHALL compare against baseline and fail if any function regresses by more than 5%.

---

### Requirement 5: End-to-End Throughput Tests

**User Story:** As a system architect, I want end-to-end throughput tests that simulate real workloads, so that I can validate the library meets production requirements.

#### Acceptance Criteria

1. WHEN running E2E tests THEN the system SHALL measure complete pipeline: read -> chunk -> hash -> compress -> sign.

2. WHEN running E2E tests THEN the system SHALL report:
   - Total throughput (MB/s)
   - Latency percentiles (p50, p95, p99, p999)
   - Chunk size distribution statistics
   - Memory high-water mark

3. WHEN running E2E tests THEN the system SHALL simulate concurrent workloads with 1, 4, 8, and 16 parallel streams.

4. WHEN running E2E tests THEN the system SHALL use realistic NAR fixtures (at least 1GB total test data).

---

### Requirement 6: Memory Allocation Tracking

**User Story:** As a performance engineer, I want to track memory allocations during benchmarks, so that I can identify and eliminate unnecessary allocations.

#### Acceptance Criteria

1. WHEN running allocation-tracking benchmarks THEN the system SHALL report:
   - Total bytes allocated
   - Number of allocations
   - Peak memory usage
   - Allocation hotspots (function/line)

2. WHEN using `CompressionScratch` THEN the system SHALL verify zero allocations after warmup phase.

3. WHEN using `ChunkStream` THEN the system SHALL verify allocation count is O(1) relative to input size (only buffer growth, not per-chunk allocations).

4. WHEN running benchmarks THEN the system SHALL integrate with `dhat` or similar allocation profiler.

---

### Requirement 7: Performance Regression Detection in CI

**User Story:** As a maintainer, I want automated performance regression detection, so that no PR can silently degrade performance.

#### Acceptance Criteria

1. WHEN a PR is opened THEN CI SHALL run the full benchmark suite.

2. WHEN benchmark results are available THEN CI SHALL compare against the baseline from the main branch.

3. IF any benchmark regresses by more than 5% THEN CI SHALL fail the PR and report the specific regressions.

4. WHEN benchmarks complete successfully THEN CI SHALL update the baseline for merged PRs.

5. WHEN reporting regressions THEN CI SHALL include flame graphs for regressed functions.

---

### Requirement 8: Zero-Length Input Handling

**User Story:** As a developer, I want all functions to handle zero-length inputs gracefully, so that edge cases do not cause panics or incorrect results.

#### Acceptance Criteria

1. WHEN `chunk_data` receives empty input THEN the system SHALL return an empty `Vec<ChunkMetadata>`.

2. WHEN `ChunkStream` receives a reader that immediately returns EOF THEN the system SHALL iterate zero times without error.

3. WHEN compression functions receive empty input THEN the system SHALL return valid compressed output that decompresses to empty.

4. WHEN hashing functions receive empty input THEN the system SHALL return the correct hash of empty data.

5. WHEN signing functions receive empty input THEN the system SHALL produce valid signatures that verify correctly.

6. WHEN base32 encode/decode receives empty input THEN the system SHALL return empty string/vec respectively.

---

### Requirement 9: Maximum Size Boundary Testing

**User Story:** As a QA engineer, I want tests that exercise maximum size boundaries, so that the library handles large inputs correctly.

#### Acceptance Criteria

1. WHEN `ChunkingOptions.max_size` is set to 1GB (the current maximum) THEN the system SHALL accept and process the configuration.

2. WHEN a chunk exactly equals `max_size` THEN the system SHALL emit it correctly without off-by-one errors.

3. WHEN input size approaches `usize::MAX` THEN the system SHALL handle overflow gracefully with appropriate errors.

4. WHEN decompression would exceed `MAX_DECOMPRESSED_SIZE` (1GB) THEN all codecs SHALL return `SizeExceeded` error.

5. WHEN buffer sizes approach system memory limits THEN the system SHALL fail gracefully with `BufferLimitExceeded` rather than OOM-killing the process.

---

### Requirement 10: Error Path Performance

**User Story:** As a performance engineer, I want error paths to be as fast as success paths, so that error handling does not become a performance bottleneck under adverse conditions.

#### Acceptance Criteria

1. WHEN invalid input is detected THEN the system SHALL return errors without allocating heap memory for error messages (use static strings or pre-allocated errors where possible).

2. WHEN errors occur during chunking THEN the system SHALL not leave partial state that requires expensive cleanup.

3. WHEN errors are propagated through the NIF layer THEN the system SHALL minimize Erlang term allocations.

4. WHEN benchmarking THEN error path latency SHALL be within 2x of success path latency for equivalent input sizes.

---

### Requirement 11: Graceful Degradation Under Memory Pressure

**User Story:** As a system operator, I want the library to degrade gracefully when system resources are constrained, so that it does not crash the entire system.

#### Acceptance Criteria

1. WHEN buffer allocation fails THEN the system SHALL return `ChunkingError::BufferLimitExceeded` instead of panicking.

2. WHEN the system is under memory pressure THEN `CompressionScratch` SHALL allow explicit capacity limits to prevent runaway growth.

3. WHEN environment variable `CHUNKER_READ_SLICE_CAP_BYTES` is set THEN the system SHALL respect the configured limit.

4. WHEN environment variable `CHUNKER_ASYNC_BUFFER_LIMIT_BYTES` is set THEN the system SHALL clamp within safe bounds (64MB-3GB).

5. IF Rayon thread pool creation fails THEN the system SHALL fall back to single-threaded operation with a warning.

---

### Requirement 12: Performance Characteristics Documentation

**User Story:** As a user of the library, I want documented performance characteristics for each function, so that I can make informed decisions about which APIs to use.

#### Acceptance Criteria

1. WHEN reading API documentation THEN each public function SHALL include:
   - Time complexity (O notation)
   - Space complexity
   - Expected throughput range (GB/s)
   - Parallelization characteristics

2. WHEN reading documentation for `chunk_data` vs `ChunkStream` THEN the tradeoffs SHALL be clearly explained (memory vs latency).

3. WHEN reading documentation for hash algorithms THEN SHA256 vs BLAKE3 performance differences SHALL be quantified.

4. WHEN reading compression documentation THEN each codec's speed/ratio tradeoffs SHALL be documented with real numbers.

---

### Requirement 13: Operator Tuning Guide

**User Story:** As a system operator, I want a tuning guide that explains how to optimize performance for my workload, so that I can maximize throughput in production.

#### Acceptance Criteria

1. WHEN reading the tuning guide THEN operators SHALL find guidance on:
   - Optimal chunk sizes for different NAR characteristics
   - Compression level selection based on CPU/bandwidth tradeoffs
   - Thread pool sizing for different core counts
   - Memory budget configuration
   - Environment variable reference

2. WHEN reading the tuning guide THEN operators SHALL find benchmark reproduction commands.

3. WHEN reading the tuning guide THEN operators SHALL find monitoring integration guidance (metrics exposition).

4. WHEN reading the tuning guide THEN operators SHALL find troubleshooting steps for common performance issues.

---

### Requirement 14: Benchmark Reproduction Instructions

**User Story:** As a contributor, I want clear instructions to reproduce benchmarks, so that I can validate my optimizations locally.

#### Acceptance Criteria

1. WHEN reading benchmark docs THEN contributors SHALL find:
   - Hardware requirements (minimum, recommended)
   - OS and dependency versions
   - Exact commands to run each benchmark suite
   - Expected baseline numbers for reference hardware

2. WHEN running benchmarks THEN the system SHALL validate the environment (CPU governor, turbo boost state, NUMA topology) and warn if non-optimal.

3. WHEN benchmarks complete THEN the system SHALL output reproducibility metadata (CPU model, memory, kernel version, Rust version).

---

### Requirement 15: Dependency Audit and Optimization

**User Story:** As a maintainer, I want minimal, optimized dependencies, so that build times are fast and binary size is small.

#### Acceptance Criteria

1. WHEN auditing dependencies THEN the team SHALL evaluate:
   - `lzma-rs` vs `xz2` for XZ compression (performance, binary size)
   - `lz4_flex` vs alternatives (pure Rust vs C bindings)
   - `bzip2` necessity (consider removal if rarely used)

2. WHEN evaluating dependencies THEN each SHALL be benchmarked against alternatives.

3. WHEN dependencies are selected THEN unused features SHALL be disabled via Cargo feature flags.

4. IF a dependency can be replaced with a lighter alternative without performance loss THEN it SHALL be replaced.

5. WHEN the audit completes THEN results SHALL be documented including rationale for each dependency choice.

---

### Requirement 16: Feature Flag Minimization

**User Story:** As a library user, I want fine-grained feature flags, so that I can include only the functionality I need and minimize binary size.

#### Acceptance Criteria

1. WHEN building the library THEN users SHALL be able to independently enable/disable:
   - Each compression codec (zstd, lz4, xz, bzip2)
   - Each hash algorithm (sha256, blake3)
   - Signing functionality
   - Async streaming support
   - NIF bindings
   - Telemetry

2. WHEN building with minimal features THEN binary size SHALL be documented and optimized.

3. WHEN a feature is disabled THEN its dependencies SHALL not be compiled.

4. WHEN default features are selected THEN they SHALL represent the common use case (chunking + zstd + sha256).

---

### Requirement 17: Static vs Dynamic Linking Analysis

**User Story:** As a distribution maintainer, I want guidance on static vs dynamic linking tradeoffs, so that I can choose the optimal configuration for my deployment.

#### Acceptance Criteria

1. WHEN documenting linking options THEN the guide SHALL cover:
   - Performance implications of static vs dynamic linking
   - Binary size differences
   - Deployment complexity tradeoffs
   - Security update considerations

2. WHEN building for NIF deployment THEN the recommended linking strategy SHALL be documented.

3. WHEN linking to system libraries (zstd, lz4) THEN the system SHALL support both static and dynamic options via feature flags.

---

### Requirement 18: x86_64 SIMD Optimizations (AVX2/AVX-512)

**User Story:** As a performance engineer, I want the library to use AVX2/AVX-512 instructions when available, so that hashing and chunking achieve maximum throughput on modern CPUs.

#### Acceptance Criteria

1. WHEN running on x86_64 with AVX2 support THEN BLAKE3 hashing SHALL use AVX2 instructions (already enabled via `blake3` crate with `rayon` feature).

2. WHEN running on x86_64 with AVX-512 support THEN the system SHALL detect and use AVX-512 for BLAKE3.

3. WHEN running on x86_64 THEN SHA256 SHALL use SHA-NI instructions if available (enabled via `sha2/asm` feature).

4. WHEN SIMD optimizations are active THEN the system SHALL log the detected CPU features at debug level.

5. WHEN running benchmarks THEN the system SHALL report which SIMD features are in use.

---

### Requirement 19: ARM NEON Optimizations

**User Story:** As a user deploying on ARM servers (AWS Graviton, Apple Silicon), I want native NEON optimizations, so that performance is competitive with x86_64.

#### Acceptance Criteria

1. WHEN running on ARM64 with NEON support THEN BLAKE3 SHALL use NEON instructions.

2. WHEN running on ARM64 THEN SHA256 SHALL use crypto extensions if available.

3. WHEN cross-compiling for ARM64 THEN the build system SHALL enable appropriate target features.

4. WHEN running on Apple Silicon THEN the system SHALL achieve at least 80% of x86_64 throughput for equivalent workloads.

---

### Requirement 20: Runtime Platform Detection

**User Story:** As a user, I want the library to automatically detect and use the best available CPU features at runtime, so that a single binary works optimally across different machines.

#### Acceptance Criteria

1. WHEN the library initializes THEN it SHALL detect available CPU features (AVX2, AVX-512, NEON, SHA-NI).

2. WHEN CPU features are detected THEN the system SHALL select optimal code paths automatically.

3. IF runtime detection is not possible for a feature THEN compile-time detection SHALL be used as fallback.

4. WHEN feature detection occurs THEN it SHALL happen once at startup, not per-operation.

5. WHEN running diagnostics THEN the system SHALL expose detected features via a public API.

---

### Requirement 21: Bounds Check Elimination Strategy

**User Story:** As a performance engineer, I want unnecessary bounds checks eliminated, so that hot loops achieve maximum throughput.

#### Acceptance Criteria

1. WHERE slice indexing occurs in hot paths THEN the code SHALL use patterns that allow LLVM to eliminate bounds checks:
   - Iterator-based access instead of indexing
   - `get_unchecked` with documented safety proofs
   - Split borrows to prove non-overlapping access

2. WHEN using `get_unchecked` or other unsafe operations THEN each occurrence SHALL have:
   - A `// SAFETY:` comment explaining why it's safe
   - A preceding bounds check or invariant that proves safety
   - A test that would catch if the invariant was violated

3. WHEN reviewing code THEN bounds check elimination opportunities SHALL be identified via compiler output analysis (`--emit=asm`).

---

### Requirement 22: Unsafe Block Audit and Safety Proofs

**User Story:** As a security reviewer, I want all unsafe blocks documented with safety proofs, so that I can verify memory safety guarantees.

#### Acceptance Criteria

1. WHEN unsafe code exists THEN each block SHALL have a `// SAFETY:` comment that:
   - States the specific unsafe operation being performed
   - Lists all preconditions required for safety
   - References where preconditions are established
   - Explains why the invariants cannot be violated

2. WHEN adding new unsafe code THEN the PR SHALL include proof that Miri passes.

3. WHEN unsafe code is found without safety documentation THEN it SHALL be flagged for review and documentation.

4. WHEN auditing is complete THEN a summary of all unsafe blocks SHALL be maintained in documentation.

---

### Requirement 23: Miri Validation in CI

**User Story:** As a maintainer, I want Miri to validate memory safety in CI, so that undefined behavior is caught before release.

#### Acceptance Criteria

1. WHEN CI runs THEN Miri SHALL execute the test suite to detect:
   - Use-after-free
   - Out-of-bounds access
   - Invalid pointer arithmetic
   - Data races (with `-Zmiri-check-number-validity`)

2. IF Miri detects undefined behavior THEN CI SHALL fail with detailed diagnostics.

3. WHEN Miri is too slow for the full suite THEN a representative subset SHALL be selected for CI with full suite running nightly.

4. WHEN new unsafe code is added THEN corresponding Miri tests SHALL be required.

---

### Requirement 24: Non-Functional Performance Targets

**User Story:** As a product owner, I want specific, measurable performance targets, so that we can objectively determine when the library is "world-class."

#### Acceptance Criteria

1. WHEN measuring chunking throughput THEN the target SHALL be >= 2 GB/s for FastCDC alone (without hashing) on modern x86_64.

2. WHEN measuring hashing throughput THEN BLAKE3 target SHALL be >= 5 GB/s on AVX2-capable hardware.

3. WHEN measuring hashing throughput THEN SHA256 target SHALL be >= 500 MB/s with SHA-NI.

4. WHEN measuring compression throughput THEN zstd level 3 target SHALL be >= 400 MB/s.

5. WHEN measuring end-to-end pipeline THEN chunk+hash+compress target SHALL be >= 300 MB/s.

6. WHEN measuring latency THEN p99 for a 1MB chunk operation SHALL be < 10ms.

7. WHEN measuring memory THEN overhead per chunk operation SHALL be < 10KB beyond input size.

---

### Requirement 25: Test Data Diversity for Benchmarks

**User Story:** As a QA engineer, I want benchmarks to use diverse, realistic test data, so that performance results reflect production behavior.

#### Acceptance Criteria

1. WHEN benchmarking THEN test data SHALL include:
   - Highly compressible data (zeros, repeated patterns)
   - Incompressible data (random bytes, encrypted)
   - Realistic NAR content (ELF binaries, shared libraries, text files)
   - Edge cases (tiny files, huge files, pathological patterns)

2. WHEN using NAR fixtures THEN they SHALL be generated from real Nix store paths.

3. WHEN benchmark results are reported THEN data characteristics SHALL be included for context.

4. WHEN comparing compression codecs THEN the same data sets SHALL be used for fair comparison.

---

### Requirement 26: Compile-Time Optimization Validation

**User Story:** As a performance engineer, I want to verify that compiler optimizations are applied correctly, so that release builds achieve maximum performance.

#### Acceptance Criteria

1. WHEN building for release THEN the following settings SHALL be validated:
   - LTO (Link-Time Optimization) enabled
   - `codegen-units = 1` for maximum optimization
   - Appropriate target-cpu (native or specific microarchitecture)

2. WHEN analyzing generated assembly THEN hot functions SHALL be inspected for:
   - Vectorization
   - Inlining
   - Bounds check elimination

3. WHEN profile-guided optimization (PGO) is feasible THEN it SHALL be documented and benchmarked.

4. WHEN binary size is a concern THEN size optimization profile SHALL be available and documented.
