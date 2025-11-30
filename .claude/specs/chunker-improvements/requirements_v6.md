# Requirements Document: Hot Path Optimizations for 100M+ Operations

## Introduction

This requirements document specifies hot path optimizations for the Chunker library (v0.1.0-beta6, Rust 2024 edition) designed to handle hundreds of millions of chunking operations. At this scale, every microsecond matters. The document identifies specific performance-critical code paths in the current implementation and defines requirements for optimization.

The analysis covers four primary hot paths:
1. **FastCDC Inner Loop** - Content-defined chunking iteration
2. **Hash Computation** - SHA256 and BLAKE3 per-chunk hashing
3. **Buffer Management** - BytesMut operations in ChunkStream
4. **Iterator Implementation** - ChunkStream::next() state machine

Each requirement targets specific lines and patterns in the source code that can be optimized for maximum throughput.

---

## Requirements

### Requirement 1: FastCDC Inner Loop Optimization

**User Story:** As a system processing 100M+ chunks, I want the FastCDC cut-point detection loop to execute with minimal CPU cycles, so that chunking throughput exceeds 2.5 GB/s on modern hardware.

#### Acceptance Criteria

1. WHEN FastCDC iterates over bytes in `ChunkStream::next()` (chunking.rs:412-452) THEN the system SHALL minimize redundant FastCDC iterator construction by caching gear tables and internal state across iterations.

2. WHEN the FastCDC library performs gear table lookups THEN the system SHALL ensure the 256-entry gear table is cache-line aligned (64-byte alignment) to prevent false sharing and maximize L1 cache hits.

3. WHEN processing cut-point detection in tight loops THEN the system SHALL avoid recreating `FastCDC::new()` on every `ChunkStream::next()` call (currently at line 412-417) by implementing a resumable chunking state.

4. IF the data buffer contains fewer bytes than `min_size` THEN the system SHALL skip the FastCDC iteration entirely to avoid unnecessary gear hash computations.

5. WHEN collecting cut_points in the loop (chunking.rs:420-452) THEN the system SHALL pre-allocate the `Vec<Chunk>` with estimated capacity based on `buffer.len() / avg_size` to eliminate mid-loop reallocations.

6. WHEN the inner loop iterates (chunking.rs:423-451) THEN the system SHALL ensure the compiler can unroll the loop by maintaining constant-time operations and avoiding data-dependent branches.

### Requirement 2: Hash Computation Hot Path

**User Story:** As a developer optimizing cryptographic operations, I want hash computations to leverage SIMD and hardware acceleration, so that per-chunk hashing overhead is minimized below 1 microsecond for typical chunk sizes.

#### Acceptance Criteria

1. WHEN computing SHA256 hashes via `sha256_hash_raw()` (hashing.rs:38-42) THEN the system SHALL use the `sha2/asm` feature (currently enabled) and verify generated assembly uses SHA-NI instructions on x86_64.

2. WHEN computing BLAKE3 hashes via `blake3::hash()` (chunking.rs:246, 494, 555, 711, 753) THEN the system SHALL ensure the `rayon` feature enables multi-threaded hashing for chunks larger than 64KB.

3. WHEN hashing chunks in parallel (chunking.rs:515-520) THEN the system SHALL batch small chunks (< 4KB) to amortize Rayon thread pool overhead, processing them sequentially instead.

4. IF a chunk is smaller than 1KB THEN the system SHALL use inline hashing without Rayon dispatch to avoid thread synchronization overhead.

5. WHEN the `HashAlgorithm` enum is matched (chunking.rs:488-495) THEN the system SHALL ensure the match statement compiles to a direct jump table, not a series of conditional branches.

6. WHEN processing the hashing pass in `chunk_data()` (chunking.rs:223-256) THEN the system SHALL support a "streaming hash" API that pipes chunker output directly to hasher input without intermediate buffering.

7. WHEN `Sha256::new()` is called per-chunk (chunking.rs:242, 489, 550, 706, 748) THEN the system SHALL consider hasher state reuse via `Sha256::reset()` to eliminate repeated initialization overhead.

### Requirement 3: Buffer Management Optimization

**User Story:** As a streaming chunker user, I want buffer operations to be zero-copy where possible, so that memory bandwidth is not the bottleneck when processing large files.

#### Acceptance Criteria

1. WHEN `BytesMut::split_to()` is called (chunking.rs:457, 542, 699, 741) THEN the system SHALL verify this operation is O(1) and does not trigger memory copies for the retained buffer portion.

2. WHEN `BytesMut::freeze()` is called to create `Bytes` handles (chunking.rs:457, 542, 699, 741) THEN the system SHALL ensure the underlying allocation is shared via atomic reference counting, not copied.

3. WHEN reserving buffer space (chunking.rs:577) THEN the system SHALL use geometric growth (2x) to minimize reallocation frequency, with a configurable upper bound.

4. WHEN `buffer.resize(start + read_size, 0)` is called (chunking.rs:582) THEN the system SHALL use `set_len()` with proper unsafe block instead of zero-initialization when the buffer will be immediately overwritten by `read()`.

5. WHEN the buffer grows beyond `max_size * 2` THEN the system SHALL implement buffer compaction to prevent unbounded memory growth in long-running streams.

6. IF `batch_data.slice(offset..offset + len)` is called (chunking.rs:481) THEN the system SHALL ensure slicing is zero-copy and returns a reference-counted view, not a new allocation.

7. WHEN processing the final EOF chunk (chunking.rs:534-567) THEN the system SHALL avoid the redundant `buffer.len()` call by caching the length value.

### Requirement 4: Iterator State Machine Optimization

**User Story:** As an iterator consumer, I want `ChunkStream::next()` to have minimal per-call overhead, so that the iterator pattern remains a zero-cost abstraction.

#### Acceptance Criteria

1. WHEN `ChunkStream::next()` is called (chunking.rs:403-603) THEN the system SHALL minimize the state machine branches by restructuring to a flat state representation.

2. WHEN checking `pending_chunks.pop_front()` (chunking.rs:404-406) THEN the system SHALL use `VecDeque::pop_front()` which is O(1) but verify the generated assembly avoids bounds checking overhead.

3. WHEN the loop checks `!self.buffer.is_empty()` (chunking.rs:410) THEN the system SHALL hoist this check outside the loop when possible to reduce per-iteration branch overhead.

4. IF the `eof` flag is true AND the buffer is empty (chunking.rs:534-539) THEN the system SHALL return `None` immediately without additional state checks.

5. WHEN updating `self.position` (chunking.rs:523, 559, 715) THEN the system SHALL use unchecked arithmetic where overflow is provably impossible to eliminate overflow checks.

6. WHEN extending `pending_chunks` with processed chunks (chunking.rs:526) THEN the system SHALL pre-allocate the VecDeque capacity to match expected chunk count.

7. WHEN handling `ErrorKind::Interrupted` (chunking.rs:595-597) THEN the system SHALL use `#[cold]` annotation or unlikely() hint to move this error path out of the hot instruction cache.

### Requirement 5: Branch Misprediction Mitigation

**User Story:** As a performance engineer, I want hot loops to minimize branch mispredictions, so that the CPU pipeline remains full and throughput is maximized.

#### Acceptance Criteria

1. WHEN checking `touches_end && !self.eof && len < self.max_size` (chunking.rs:446) THEN the system SHALL reorder conditions by likelihood (most likely to short-circuit first) and use `#[likely]`/`#[unlikely]` hints.

2. WHEN the parallel processing threshold `cut_points.len() > 4` is checked (chunking.rs:515) THEN the system SHALL make this threshold configurable and profile-guided to match actual workload characteristics.

3. WHEN matching `HashAlgorithm` variants (chunking.rs:240-247, 488-495) THEN the system SHALL use a function pointer table or trait object dispatch to eliminate branch instructions in tight loops.

4. WHEN checking `len == 0` for zero-length chunk detection (chunking.rs:428) THEN the system SHALL use `debug_assert!` in release builds if FastCDC is known to never produce zero-length chunks.

5. WHEN the `tracing::enabled!` check occurs (chunking.rs:499-505, 717-721) THEN the system SHALL ensure this check compiles to a single load from a static boolean, not a function call.

6. WHEN processing the cut_points loop (chunking.rs:423-452) THEN the system SHALL consider converting the `if touches_end && !self.eof && len < self.max_size { break }` pattern to a branchless computation where possible.

### Requirement 6: SIMD Vectorization Opportunities

**User Story:** As a high-performance computing user, I want data processing loops to leverage SIMD instructions, so that throughput scales with CPU vector width.

#### Acceptance Criteria

1. WHEN the nix_base32_encode function processes chunks of 5 bytes (hashing.rs:58-72) THEN the system SHALL provide an AVX2/NEON-accelerated variant for encoding 32+ bytes at once.

2. WHEN the nix_base32_decode function iterates over 8-byte chunks (hashing.rs:86-101) THEN the system SHALL use SIMD gather/scatter operations for parallel character-to-index lookup.

3. WHEN BLAKE3 is used for hashing THEN the system SHALL verify the blake3 crate's `rayon` feature enables SIMD acceleration and multi-threaded tree hashing.

4. WHEN the gear hash table lookup occurs in FastCDC THEN the system SHALL investigate SIMD gather instructions for parallel table lookups (processing 4-8 bytes simultaneously).

5. IF the target architecture supports AVX-512 THEN the system SHALL provide a feature flag to enable 512-bit vectorized implementations.

6. WHEN compressing with LZ4 via `lz4_flex` (compression.rs) THEN the system SHALL verify the crate uses SIMD-accelerated matching and literal copying.

### Requirement 7: Memory Access Pattern Optimization

**User Story:** As a cache-conscious developer, I want data structures and access patterns to maximize cache efficiency, so that memory latency does not dominate execution time.

#### Acceptance Criteria

1. WHEN `ChunkStream` struct fields are accessed (chunking.rs:270-280) THEN the system SHALL order fields by access frequency (hot fields first) and ensure the struct fits within 2 cache lines (128 bytes).

2. WHEN the gear table is accessed during FastCDC iteration THEN the system SHALL ensure prefetch hints are generated for sequential access patterns.

3. WHEN processing chunks in parallel with Rayon (chunking.rs:224-256, 517) THEN the system SHALL ensure each thread's working set fits within L2 cache to prevent cache thrashing.

4. WHEN `batch_data` is accessed via `.slice()` (chunking.rs:481) THEN the system SHALL ensure contiguous memory access to leverage hardware prefetching.

5. WHEN the `pending_chunks` VecDeque is used (chunking.rs:279, 404, 526) THEN the system SHALL verify VecDeque's ring buffer implementation provides cache-friendly sequential access.

6. WHEN reading from the input stream (chunking.rs:584) THEN the system SHALL align read sizes to page boundaries (4KB) for optimal OS buffer management.

### Requirement 8: Instruction-Level Parallelism Enhancement

**User Story:** As a CPU pipeline optimizer, I want independent operations to be schedulable in parallel, so that instruction throughput approaches theoretical maximum.

#### Acceptance Criteria

1. WHEN computing chunk metadata (chunking.rs:507-512) THEN the system SHALL break data dependencies by computing hash, offset, and length independently before combining into the struct.

2. WHEN the loop updates multiple counters (chunking.rs:421-422, 484-486) THEN the system SHALL batch counter updates to reduce instruction dependencies.

3. WHEN metrics are recorded via `counter!` and `histogram!` macros (chunking.rs:484-486, 545-547, 701-703, 743-745) THEN the system SHALL make metrics recording optional via feature flag to eliminate atomic operations in latency-critical paths.

4. WHEN the `TRACE_SAMPLE_COUNTER` atomic is accessed (chunking.rs:501-502, 718) THEN the system SHALL use `Ordering::Relaxed` (already used) and consider removing the modulo operation in favor of bit masking.

5. WHEN constructing `ChunkMetadata` (chunking.rs:507-512, 561-566, 723-728, 759-764) THEN the system SHALL ensure field initialization order allows maximum instruction overlap.

6. WHEN the `process_chunk` closure captures variables (chunking.rs:465-513) THEN the system SHALL verify the closure is inlined and captures by reference, not by move with cloning.

### Requirement 9: Zero-Cost Abstraction Verification

**User Story:** As a Rust developer, I want to verify that iterator and closure patterns compile to optimal machine code, so that abstractions do not introduce runtime overhead.

#### Acceptance Criteria

1. WHEN `ChunkStream` implements `Iterator` (chunking.rs:399-604) THEN the system SHALL verify via assembly inspection that `next()` inlines into consuming loops when optimizations are enabled.

2. WHEN closures like `process_chunk` are used (chunking.rs:465-513) THEN the system SHALL verify the closure is monomorphized and fully inlined at each call site.

3. WHEN `par_iter().map().collect()` is used (chunking.rs:224-256, 517) THEN the system SHALL verify Rayon's work-stealing does not introduce unnecessary synchronization for small workloads.

4. WHEN generic type parameters are used (`R: Read`) THEN the system SHALL verify monomorphization produces optimal code for common reader types (Cursor, BufReader, File).

5. WHEN `Option<Result<ChunkMetadata, ChunkingError>>` is returned (chunking.rs:400) THEN the system SHALL verify the nested enum compiles to efficient tagged union representation.

6. WHEN `match` expressions handle `HashAlgorithm` (chunking.rs:240-247) THEN the system SHALL verify the compiler generates a jump table or direct call, not cascading comparisons.

### Requirement 10: Compiler Hint Integration

**User Story:** As a performance-focused developer, I want to provide hints to the compiler for optimization decisions, so that generated code matches performance expectations.

#### Acceptance Criteria

1. WHEN hot functions like `ChunkStream::next()` are defined THEN the system SHALL add `#[inline]` or `#[inline(always)]` attributes where beneficial.

2. WHEN error paths are handled (chunking.rs:429-430, 435-439, 594-599) THEN the system SHALL annotate error branches with `#[cold]` to improve instruction cache locality.

3. WHEN bounds checks occur on slices (chunking.rs:238, 470-479) THEN the system SHALL use `get_unchecked()` with safety comments where bounds are proven valid.

4. WHEN loop iteration counts are bounded THEN the system SHALL add loop unrolling hints via `#[unroll]` attribute where appropriate.

5. WHEN the profile configuration is set (Cargo.toml:26-35) THEN the system SHALL ensure `lto = true` and `codegen-units = 1` are applied to maximize cross-module optimization.

6. WHEN feature flags like `asm` are defined (Cargo.toml:42) THEN the system SHALL document the performance impact and ensure CI benchmarks cover both enabled and disabled states.

### Requirement 11: Metrics and Tracing Overhead Reduction

**User Story:** As a production operator, I want observability to have negligible performance impact, so that monitoring does not degrade throughput.

#### Acceptance Criteria

1. WHEN `counter!` and `histogram!` macros are called in hot paths (chunking.rs:484-486) THEN the system SHALL provide a `no-metrics` feature flag to compile out all metrics code paths.

2. WHEN `tracing::enabled!(tracing::Level::TRACE)` is checked (chunking.rs:499) THEN the system SHALL ensure this compiles to a single static boolean load when tracing is disabled.

3. WHEN the `TRACE_SAMPLE_EVERY` constant is defined (chunking.rs:283) THEN the system SHALL use a power-of-two value (1024 is good) to enable bit-mask sampling instead of modulo.

4. IF tracing is disabled at compile time THEN the system SHALL eliminate all tracing-related code from the hot path via `#[cfg]` attributes.

5. WHEN `instrument` attribute macros are used (chunking.rs:136, 186, 331, 370) THEN the system SHALL ensure these are on cold/setup paths only, not on per-chunk hot paths.

6. WHEN debug logging occurs THEN the system SHALL ensure `debug!` and `trace!` macros compile to no-ops when the `release` profile is active and tracing feature is disabled.

### Requirement 12: Parallelization Threshold Tuning

**User Story:** As a workload optimizer, I want parallelization decisions to be based on measured workload characteristics, so that overhead is minimized for small batches.

#### Acceptance Criteria

1. WHEN the parallel threshold `cut_points.len() > 4` is checked (chunking.rs:515) THEN the system SHALL make this threshold runtime-configurable via environment variable or builder pattern.

2. WHEN Rayon parallel iteration is used THEN the system SHALL consider chunk data size, not just chunk count, when deciding parallelization (e.g., total bytes > 64KB).

3. WHEN the thread pool is accessed THEN the system SHALL minimize thread pool contention by batching work items into coarser units.

4. IF chunks are consistently small (< 4KB average) THEN the system SHALL skip parallel hashing entirely and process sequentially.

5. WHEN `chunk_data()` is called (chunking.rs:187-194) THEN the system SHALL provide an option to disable Rayon parallelization for single-threaded use cases.

6. WHEN the Rayon thread pool is initialized THEN the system SHALL verify it uses the global pool and does not create per-call thread pools.

---

## Non-Functional Requirements

### NFR-1: Performance Targets

1. The optimized implementation SHALL achieve > 2.5 GB/s raw FastCDC throughput on x86_64 with AVX2.
2. The SHA256 hashing with ASM feature SHALL achieve > 1 GB/s on modern Intel/AMD CPUs with SHA-NI.
3. The BLAKE3 hashing SHALL achieve > 3 GB/s on multi-core systems with the rayon feature.
4. The ChunkStream iterator overhead SHALL be < 100ns per `next()` call beyond underlying operations.
5. Memory allocation overhead SHALL not exceed 5% of total processing time.

### NFR-2: Benchmark Requirements

1. The project SHALL maintain criterion benchmarks for all hot paths.
2. Benchmarks SHALL be run on CI with consistent hardware profiles.
3. Performance regressions > 5% SHALL fail CI builds.
4. Assembly output for critical functions SHALL be reviewed during code review.

### NFR-3: Compatibility Requirements

1. Optimizations SHALL not break existing public API contracts.
2. SIMD optimizations SHALL gracefully fall back on unsupported architectures.
3. Feature flags SHALL allow disabling optimizations that increase binary size.
4. The NIF interface SHALL remain compatible with Erlang VM scheduling requirements.

---

## Appendix: Hot Path Code Locations

| Hot Path | File | Lines | Current Issue |
|----------|------|-------|---------------|
| FastCDC construction | chunking.rs | 412-417 | Recreated per next() call |
| Gear table lookup | fastcdc crate | external | Not cache-aligned by default |
| SHA256 hashing | chunking.rs | 242-244, 489-492, 550-553 | Per-chunk hasher allocation |
| BLAKE3 hashing | chunking.rs | 246, 494, 555 | No batching for small chunks |
| Buffer split_to | chunking.rs | 457, 542, 699, 741 | Verify zero-copy |
| Buffer resize | chunking.rs | 582 | Zero-initialization overhead |
| Parallel threshold | chunking.rs | 515 | Fixed at 4 chunks |
| Metrics recording | chunking.rs | 484-486 | Atomic ops in hot path |
| Tracing check | chunking.rs | 499-505 | Sampled but still overhead |
| Iterator state | chunking.rs | 403-603 | Multiple branch points |
