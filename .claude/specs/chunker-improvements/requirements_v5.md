# Requirements Document: Hot Path Optimizations

## Introduction

This document specifies requirements for optimizing the hot paths in the Chunker library that processes hundreds of millions of chunking operations. At this scale, every microsecond matters. The primary focus is on micro-optimizations, allocation elimination, cache efficiency, function call overhead reduction, metrics overhead mitigation, and lock contention prevention.

The Chunker library (v0.1.0-beta6, Rust 2024 edition) uses FastCDC for content-defined chunking with BLAKE3/SHA256 hashing, Zstd/LZ4/XZ compression, and integrates with metrics collection. The identified hot paths are:

1. **ChunkStream::next()** (chunking.rs:403-603) - THE hottest path, called for every chunk
2. **chunk_data_with_hash()** (chunking.rs:201-262) - In-memory parallel chunking
3. **sha256_hash_raw() / blake3_hash()** (hashing.rs:38-50) - Per-chunk hashing
4. **compress_zstd() / decompress_auto()** (compression.rs) - Per-chunk compression

---

## Requirements

### Requirement 1: Inline Critical Hot Path Functions

**User Story:** As a systems engineer processing 100M+ chunks, I want critical hot path functions to be inlined, so that function call overhead is eliminated for maximum throughput.

#### Acceptance Criteria

1. WHEN the `ChunkStream::next()` method is called THEN the system SHALL ensure the inner `process_chunk` closure (chunking.rs:465-513) is compiled with `#[inline(always)]` semantics to eliminate closure call overhead.

2. WHEN `sha256_hash_raw()` (hashing.rs:38-42) is invoked THEN the system SHALL mark this function with `#[inline]` to enable cross-crate inlining for per-chunk hash computations.

3. WHEN `blake3_hash()` (hashing.rs:47-50) is invoked THEN the system SHALL mark this function with `#[inline]` to enable cross-crate inlining.

4. WHEN the `HashAlgorithm` match expression (chunking.rs:488-495, 549-556) is evaluated THEN the compiler SHALL be able to monomorphize the hash selection at compile time through generic parameterization or const generics.

5. WHEN `decompress_auto_into()` (compression.rs:216-248) performs magic byte detection THEN the system SHALL mark the function with `#[inline]` to eliminate function call overhead in the hot decompression path.

---

### Requirement 2: Eliminate Heap Allocations in Hot Paths

**User Story:** As a performance engineer, I want zero heap allocations in the chunk iteration hot path, so that memory allocation overhead does not degrade throughput at scale.

#### Acceptance Criteria

1. WHEN `ChunkStream::next()` creates a `Vec<_>` for `cut_points` (chunking.rs:420) THEN the system SHALL use a stack-allocated `SmallVec` or `ArrayVec` with capacity matching typical batch sizes (8-16 chunks) to avoid heap allocation for common cases.

2. WHEN `ChunkStream::new_with_hash()` initializes `pending_chunks: VecDeque::new()` (chunking.rs:394) THEN the system SHALL pre-allocate the VecDeque with `VecDeque::with_capacity()` matching the expected batch size to avoid reallocations during iteration.

3. WHEN the `process_chunk` closure (chunking.rs:465-513) constructs `ChunkMetadata` THEN the system SHALL ensure `Bytes::slice()` (line 481) performs zero-copy reference counting without allocation.

4. WHEN `chunk_data_with_hash()` calls `Bytes::copy_from_slice()` (chunking.rs:253) THEN the system SHALL evaluate using `Bytes::from()` with owned data or zero-copy slicing to eliminate the copy when the source data lifetime permits.

5. WHEN `effective_read_slice_cap()` (chunking.rs:294-300) or `effective_async_buffer_limit()` (chunking.rs:303-312) are called THEN the system SHALL cache the environment variable lookup result in a `OnceLock<usize>` static to avoid repeated `env::var()` allocations per chunk stream creation.

6. WHEN error paths construct error messages with `format!()` (chunking.rs:437-438, 684-687) THEN the system SHALL use static string references or pre-allocated error variants to avoid heap allocation on error paths that may still be in the hot path.

---

### Requirement 3: Optimize Branch Prediction and Cold Paths

**User Story:** As a systems programmer, I want branch prediction hints on unlikely code paths, so that the CPU pipeline is optimized for the common case.

#### Acceptance Criteria

1. WHEN `ChunkStream::next()` checks `chunk.length == 0` (chunking.rs:428) THEN the system SHALL annotate this branch as `#[cold]` or use `std::intrinsics::unlikely()` (via `likely` crate) since zero-length chunks are exceptional.

2. WHEN `ChunkStream::next()` checks `cut_points.is_empty() && offset != 0` (chunking.rs:434) THEN the system SHALL mark this validation branch as cold/unlikely since it represents an invariant violation.

3. WHEN `decompress_auto_into()` (compression.rs:216-248) performs format detection THEN the system SHALL order magic byte checks by frequency (Zstd most common first) and mark the `UnknownFormat` error path as `#[cold]`.

4. WHEN `ChunkStream::next()` handles `ErrorKind::Interrupted` (chunking.rs:595-598) THEN the system SHALL mark this retry path as unlikely since interrupts are rare in normal operation.

5. WHEN bounds checking fails in `process_chunk` (chunking.rs:470-479) THEN the system SHALL mark the error return path as `#[cold]` to optimize the common success path.

---

### Requirement 4: Reduce Metrics Collection Overhead

**User Story:** As a high-throughput service operator, I want metrics collection to have minimal impact on chunking performance, so that observability does not compromise throughput at 100M+ operations.

#### Acceptance Criteria

1. WHEN `ChunkStream::next()` records metrics (chunking.rs:484-486, 545-547, 701-703, 742-745) with 3 metric calls per chunk THEN the system SHALL implement batched metrics aggregation using thread-local counters that flush periodically rather than per-chunk atomic operations.

2. WHEN metrics are recorded in the hot path THEN the system SHALL provide a compile-time feature flag (`metrics-hot-path`) that allows complete elimination of hot-path metrics calls in production builds where maximum performance is required.

3. WHEN `counter!()` and `histogram!()` macros are invoked THEN the system SHALL evaluate using `metrics::Counter::increment()` with cached counter handles rather than repeated macro-based lookups to avoid hash map lookups per call.

4. WHEN recording `chunker.bytes_processed` (chunking.rs:485, 546, 702, 744) THEN the system SHALL batch byte counts across chunks within a single `next()` call iteration rather than incrementing per-chunk.

5. WHEN the tracing sample check `TRACE_SAMPLE_COUNTER.fetch_add(1, Ordering::Relaxed)` (chunking.rs:500-505, 717-721) is performed THEN the system SHALL use `Ordering::Relaxed` (already correct) but evaluate moving the sample check outside the parallel iteration to avoid contention.

6. IF metrics are enabled AND chunk throughput exceeds 1M chunks/second THEN the system SHALL automatically switch to sampled metrics (e.g., record every 1024th chunk) to prevent metrics from becoming the bottleneck.

---

### Requirement 5: Optimize Parallel Processing Threshold

**User Story:** As a performance engineer, I want the parallel vs. sequential processing decision to be optimized for actual workloads, so that Rayon overhead is not incurred for small batches.

#### Acceptance Criteria

1. WHEN `ChunkStream::next()` decides between `par_iter()` and `iter()` (chunking.rs:515-520) based on `cut_points.len() > 4` THEN the system SHALL make this threshold configurable and benchmark-derived, considering that Rayon spawn overhead may exceed sequential cost for small data sizes.

2. WHEN `chunk_data_with_hash()` uses unconditional `par_iter()` (chunking.rs:223-256) THEN the system SHALL add a conditional path that uses sequential iteration when chunk count is below a configurable threshold (suggested: 8 chunks).

3. WHEN parallel iteration is chosen THEN the system SHALL evaluate using `par_chunks()` or `par_bridge()` with explicit chunk sizing to optimize cache locality and reduce Rayon scheduling overhead.

4. WHEN the number of available Rayon threads is 1 (single-core environment) THEN the system SHALL bypass Rayon entirely and use sequential iteration to avoid unnecessary abstraction overhead.

5. IF parallel processing is selected AND the average chunk size is less than 64KB THEN the system SHALL increase the parallel threshold since hashing overhead is lower for small chunks and parallelism benefits are reduced.

---

### Requirement 6: Improve Cache Efficiency and Data Locality

**User Story:** As a systems engineer, I want data structures to be cache-friendly and access patterns to maximize cache hits, so that memory bandwidth is used efficiently.

#### Acceptance Criteria

1. WHEN `ChunkMetadata` (chunking.rs:62-75) is stored and accessed THEN the system SHALL evaluate reordering fields for optimal alignment: `hash: [u8; 32]` (32 bytes), `offset: u64` (8 bytes), `length: usize` (8 bytes), `payload: Bytes` (24 bytes on 64-bit) - total 72 bytes, should consider padding to 64 or 128 byte cache line boundaries.

2. WHEN `ChunkStream` (chunking.rs:270-280) is instantiated THEN the system SHALL evaluate using `#[repr(C)]` or `#[repr(align(64))]` on the struct to ensure cache line alignment and prevent false sharing when multiple streams operate concurrently.

3. WHEN `FastCDC::new()` (chunking.rs:412-417) is called repeatedly in the `next()` loop THEN the system SHALL evaluate caching the FastCDC configuration parameters to avoid repeated struct construction.

4. WHEN processing chunks in parallel (chunking.rs:517) THEN the system SHALL ensure each thread operates on cache-line-aligned boundaries to prevent false sharing on the shared `batch_data` Bytes reference counter.

5. WHEN the `buffer: BytesMut` (chunking.rs:272) is accessed THEN the system SHALL ensure read operations use sequential access patterns and consider prefetch hints for large buffer scans.

6. WHEN `pending_chunks: VecDeque` (chunking.rs:279) is accessed THEN the system SHALL evaluate using a simple `Vec` with index tracking instead, as VecDeque has pointer indirection overhead that may impact cache performance.

---

### Requirement 7: Eliminate Dynamic Dispatch in Hot Paths

**User Story:** As a compiler optimization engineer, I want hot paths to use static dispatch exclusively, so that virtual table lookups and indirect calls are eliminated.

#### Acceptance Criteria

1. WHEN `AutoDecompressReader` (compression.rs:264-343) uses `inner: Box<dyn Read + Send>` THEN the system SHALL provide an alternative enum-based implementation that enables static dispatch for the common compression formats.

2. WHEN `HashAlgorithm` enum (chunking.rs:54-58) is matched in hot paths THEN the system SHALL evaluate using const generics or separate specialized functions to enable monomorphization and eliminate runtime branching.

3. WHEN `ChunkStream<R: Read>` (chunking.rs:270) is generic over `R` THEN the system SHALL ensure the `Read` trait calls are monomorphized by avoiding `&mut dyn Read` patterns internally.

4. WHEN `CompressionStrategy` (compression.rs:189-194) is matched in `compress()` (compression.rs:202-208) THEN the system SHALL inline the match or use const generics to eliminate runtime dispatch overhead.

5. IF a hot path function takes a trait object parameter THEN the system SHALL provide a generic alternative that accepts `impl Trait` for zero-cost abstraction.

---

### Requirement 8: Optimize Hash Computation

**User Story:** As a cryptography performance engineer, I want hash computations to leverage SIMD and hardware acceleration, so that per-chunk hashing is as fast as possible.

#### Acceptance Criteria

1. WHEN SHA-256 hashing is performed via `sha256_hash_raw()` (hashing.rs:38-42) THEN the system SHALL ensure the `sha2/asm` feature is enabled (already configured in Cargo.toml) and verify SIMD utilization through benchmarks.

2. WHEN BLAKE3 hashing is performed via `blake3::hash()` (hashing.rs:48, chunking.rs:246, 494, 555, 711, 753) THEN the system SHALL ensure the `blake3/rayon` feature (already enabled) is leveraged for large chunks exceeding the BLAKE3 internal parallelism threshold.

3. WHEN creating a `Sha256` hasher instance (hashing.rs:39, chunking.rs:242, 490, 551, 707, 749) THEN the system SHALL evaluate using a thread-local hasher pool to avoid repeated initialization overhead, as hasher creation involves internal state setup.

4. WHEN hashing chunk data THEN the system SHALL ensure the data pointer is aligned to 16 or 32 byte boundaries when possible to maximize SIMD throughput.

5. WHEN parallel hashing is performed in `chunk_data_with_hash()` (chunking.rs:223-256) THEN the system SHALL ensure each thread has its own hasher instance to avoid any synchronization overhead.

---

### Requirement 9: Reduce Lock Contention

**User Story:** As a concurrent systems engineer, I want the chunking library to minimize lock contention, so that parallel processing scales linearly with core count.

#### Acceptance Criteria

1. WHEN `TRACE_SAMPLE_COUNTER: AtomicU64` (chunking.rs:282) is incremented in parallel contexts THEN the system SHALL evaluate using thread-local counters with periodic aggregation to eliminate atomic contention.

2. WHEN Rayon's thread pool executes parallel hashing (chunking.rs:517, 224) THEN the system SHALL ensure no shared mutable state requires synchronization during the parallel operation.

3. WHEN metrics counters are incremented (chunking.rs:484-486) THEN the system SHALL use thread-local aggregation buffers that flush to global counters at batch boundaries rather than per-operation atomic increments.

4. WHEN multiple `ChunkStream` instances operate concurrently THEN the system SHALL ensure they share no mutable static state that could cause contention.

5. IF `spawn_hashing_worker()` (hashing.rs:126-155) or `spawn_zstd_worker()` (compression.rs:759-780) are used THEN the system SHALL ensure the bounded channels have sufficient capacity to avoid blocking contention under burst loads.

---

### Requirement 10: Optimize Buffer Management

**User Story:** As a memory efficiency engineer, I want buffer operations to be optimized for the chunking access patterns, so that memory bandwidth is maximized.

#### Acceptance Criteria

1. WHEN `BytesMut::reserve()` and `resize()` are called (chunking.rs:577, 582) THEN the system SHALL use exponential growth with a configurable cap to reduce reallocation frequency while bounding memory usage.

2. WHEN `buffer.split_to()` is called (chunking.rs:457, 542, 699, 741) THEN the system SHALL ensure this operation is O(1) as designed by bytes crate, and verify no hidden copies occur.

3. WHEN `self.buffer.truncate(start)` is called after failed reads (chunking.rs:587, 596) THEN the system SHALL verify this does not deallocate memory, preserving capacity for future reads.

4. WHEN `Vec::with_capacity()` is used (chunking.rs:948, compression.rs:516, 609, 685) THEN the system SHALL use accurate capacity estimates to avoid both over-allocation and reallocation.

5. WHEN `CompressionScratch` (compression.rs:26-175) reuses buffers THEN the system SHALL ensure the `clear()` operation does not deallocate and capacity is preserved across operations.

6. WHEN processing large files with `ChunkStream` THEN the system SHALL implement a buffer size adaptation strategy that learns from read patterns to optimize the target read size.

---

### Requirement 11: Optimize Compression Hot Paths

**User Story:** As a compression engineer, I want compression operations to minimize overhead and maximize throughput, so that per-chunk compression does not become a bottleneck.

#### Acceptance Criteria

1. WHEN `compress_zstd_into_internal()` (compression.rs:537-561) creates a new encoder THEN the system SHALL evaluate using a thread-local encoder pool to reuse zstd context state and avoid repeated initialization.

2. WHEN `decompress_auto_into()` (compression.rs:216-248) performs magic byte detection THEN the system SHALL use a branchless comparison or lookup table for the 4-byte magic sequences to optimize the common path.

3. WHEN LZ4 compression is performed (compression.rs:434-448, 455-466) THEN the system SHALL use the frame encoder with pre-sized output buffers to minimize internal reallocations.

4. WHEN `compress_lz4_into()` (compression.rs:455-466) uses `std::mem::take()` on the output buffer THEN the system SHALL evaluate whether this pattern causes unnecessary capacity loss.

5. WHEN decompression readers are created with `.take(MAX_DECOMPRESSED_SIZE)` (compression.rs:297-333, 389-414, 477, 707) THEN the system SHALL verify this does not add measurable overhead in the hot path.

---

### Requirement 12: Provide Benchmarking Infrastructure for Validation

**User Story:** As a performance validation engineer, I want comprehensive micro-benchmarks for all hot paths, so that optimization efforts can be measured and validated.

#### Acceptance Criteria

1. WHEN optimizing hot paths THEN the system SHALL provide criterion benchmarks that measure:
   - `ChunkStream::next()` iterations per second with various buffer sizes
   - Hash computation throughput (GB/s) for both SHA-256 and BLAKE3
   - Parallel vs. sequential threshold crossover points
   - Metrics overhead with and without instrumentation

2. WHEN running benchmarks THEN the system SHALL include memory allocation profiling using `dhat` or `heaptrack` to verify zero-allocation claims.

3. WHEN benchmarking parallel operations THEN the system SHALL measure scaling efficiency across 1, 2, 4, 8, and 16 cores to validate contention-free design.

4. WHEN benchmarking THEN the system SHALL use realistic data patterns including:
   - Highly compressible data (all zeros)
   - Incompressible random data
   - Mixed content (simulating real NAR files)

5. WHEN benchmarks complete THEN the system SHALL report:
   - Operations per second
   - Bytes processed per second
   - 99th percentile latency
   - Memory allocations per operation

---

## Non-Functional Requirements

### NFR-1: Performance Targets

1. WHEN processing 100 million chunks THEN the system SHALL complete in under 10 minutes on an 8-core system with 32GB RAM.

2. WHEN `ChunkStream::next()` is invoked THEN the average latency SHALL be under 1 microsecond for chunks under 1MB (excluding I/O wait).

3. WHEN metrics are enabled THEN the overhead SHALL be less than 5% of total processing time.

4. WHEN parallel hashing is enabled THEN the throughput SHALL scale to at least 80% efficiency on 8 cores.

### NFR-2: Memory Efficiency

1. WHEN processing chunks THEN the hot path SHALL perform zero heap allocations for the common case (batch size <= 16 chunks).

2. WHEN `ChunkStream` is instantiated THEN the initial memory footprint SHALL be under 128KB before data is read.

3. WHEN metrics aggregation is enabled THEN the memory overhead per thread SHALL be under 4KB.

### NFR-3: Backward Compatibility

1. WHEN optimizations are applied THEN the public API SHALL remain unchanged.

2. WHEN new feature flags are added THEN existing code without the flags SHALL behave identically to the current implementation.

3. WHEN internal data structure layouts change THEN serialization formats (serde) SHALL remain compatible.

### NFR-4: Compile-Time Optimization

1. WHEN building in release mode THEN LTO (Link-Time Optimization) SHALL be enabled (already configured).

2. WHEN optimizations require nightly features THEN the system SHALL provide stable-compatible fallbacks.

3. WHEN profile-guided optimization (PGO) is available THEN the build system SHALL support PGO builds for maximum performance.

---

## Appendix: Identified Hot Path Code Locations

### ChunkStream::next() Critical Sections

| Line Range | Operation | Optimization Opportunity |
|------------|-----------|-------------------------|
| 403-406 | VecDeque::pop_front | Consider Vec with index |
| 412-417 | FastCDC::new | Cache configuration |
| 420 | Vec::new for cut_points | Use SmallVec |
| 484-486 | 3x metrics calls | Batch aggregation |
| 488-495 | Hash algorithm match | Monomorphization |
| 500-505 | Atomic sample counter | Thread-local counter |
| 515-520 | par_iter threshold | Tune threshold |
| 577-582 | Buffer reserve/resize | Exponential growth |

### Hashing Critical Sections

| Line | Operation | Optimization Opportunity |
|------|-----------|-------------------------|
| hashing.rs:39-41 | Sha256::new() + update + finalize | Hasher pooling |
| hashing.rs:48-49 | blake3::hash() | Already optimized |

### Compression Critical Sections

| Line Range | Operation | Optimization Opportunity |
|------------|-----------|-------------------------|
| 216-248 | Magic byte detection | Branchless comparison |
| 537-561 | Zstd encoder creation | Encoder pooling |
| 455-466 | LZ4 compression | Buffer pre-sizing |
