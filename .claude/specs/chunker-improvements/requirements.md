# Consolidated Requirements: World-Class Performance for Chunker Library

## Introduction

This document consolidates all requirements for optimizing the Chunker library to achieve world-class performance capable of processing hundreds of millions of chunking operations. The library is a high-performance content-defined chunking system for Nix NARs, written in Rust 2024 edition (v0.1.0-beta6).

**Performance Philosophy:** Every millisecond matters at scale. Every microsecond matters in hot paths. Zero-copy, zero-allocation, zero-overhead.

**Target Performance Characteristics:**
- FastCDC chunking: >= 2.5 GB/s
- BLAKE3 hashing: >= 3-5 GB/s (multi-core with SIMD)
- SHA256 hashing: >= 1 GB/s (with SHA-NI)
- Zstd compression (level 3): >= 400-500 MB/s
- End-to-end pipeline: >= 300 MB/s
- Per-chunk latency: < 1 microsecond (excluding I/O)
- p99 latency for 1MB chunk: < 5-10ms

---

## 1. Critical Bug Fixes

### BUG-001: Bzip2 Decompression Bomb Detection [P0]

**Location:** `compression.rs:714`

**Description:** The bzip2 decompression bomb check uses `==` instead of `>=` comparison, failing to detect bombs that decompress to exactly `MAX_DECOMPRESSED_SIZE`.

**Current Code:**
```rust
if (output.len() - start_len) as u64 == MAX_DECOMPRESSED_SIZE
```

**Acceptance Criteria:**
1. WHEN the bzip2 decompressor reads exactly `MAX_DECOMPRESSED_SIZE` bytes THEN the system SHALL continue reading to detect if additional data exists beyond the limit.
2. WHEN bzip2 decompression output exceeds `MAX_DECOMPRESSED_SIZE` THEN the system SHALL return `CompressionError::SizeExceeded`.
3. IF a legitimate file decompresses to exactly `MAX_DECOMPRESSED_SIZE` bytes THEN the system SHALL successfully decompress it without false-positive rejection.
4. WHEN implementing the fix THEN the system SHALL use the same `take(limit + 1)` pattern used in `decompress_zstd_into_with_limit` and `decompress_lz4_into` functions.

---

### BUG-002: Integer Overflow in FastCDC Size Casting [P0]

**Location:** `chunking.rs:214-216`

**Description:** The casts `options.min_size as u32`, `options.avg_size as u32`, `options.max_size as u32` can overflow if sizes exceed `u32::MAX`.

**Acceptance Criteria:**
1. WHEN `ChunkingOptions` values (min_size, avg_size, max_size) are cast from `usize` to `u32` for FastCDC initialization THEN the system SHALL validate that values fit within u32 range before casting.
2. IF any chunking size parameter exceeds `u32::MAX` THEN the system SHALL return `ChunkingError::InvalidOptions` with a descriptive message.
3. WHEN the `ChunkingOptions::validate()` method runs THEN it SHALL include overflow checks for u32 compatibility.
4. WHILE processing chunk definitions from FastCDC THEN the system SHALL use checked arithmetic for offset + length calculations to prevent overflow.
5. WHEN casting between integer types THEN the system SHALL use `TryFrom` or checked conversions with explicit bounds checking.

---

## 2. Hot Path Optimizations

### HOT-001: FastCDC Construction Caching [P0]

**Location:** `chunking.rs:412-417`

**Description:** `FastCDC::new()` is recreated on every `ChunkStream::next()` call, wasting CPU cycles.

**Acceptance Criteria:**
1. WHEN `ChunkStream::next()` is called THEN the system SHALL cache FastCDC configuration parameters to avoid repeated struct construction.
2. WHEN FastCDC iterates over bytes THEN the system SHALL minimize redundant iterator construction by caching gear tables and internal state across iterations.
3. WHEN processing cut-point detection THEN the system SHALL implement a resumable chunking state.
4. IF the data buffer contains fewer bytes than `min_size` THEN the system SHALL skip the FastCDC iteration entirely.

---

### HOT-002: Inline Critical Hot Path Functions [P0]

**Location:** `chunking.rs:465-513`, `hashing.rs:38-50`

**Description:** Critical functions lack inlining hints, causing function call overhead.

**Acceptance Criteria:**
1. WHEN the `process_chunk` closure is called THEN the system SHALL ensure it is compiled with `#[inline(always)]` semantics.
2. WHEN `sha256_hash_raw()` is invoked THEN the system SHALL mark this function with `#[inline]`.
3. WHEN `blake3_hash()` is invoked THEN the system SHALL mark this function with `#[inline]`.
4. WHEN the `HashAlgorithm` match expression is evaluated THEN the compiler SHALL monomorphize the hash selection at compile time.
5. WHEN `decompress_auto_into()` performs magic byte detection THEN the system SHALL mark the function with `#[inline]`.

---

### HOT-003: Branch Prediction Optimization [P1]

**Location:** `chunking.rs:428-439, 446, 594-599`

**Description:** Hot paths contain branches without prediction hints.

**Acceptance Criteria:**
1. WHEN `ChunkStream::next()` checks `chunk.length == 0` THEN the system SHALL annotate this branch as `#[cold]` or use `unlikely()`.
2. WHEN `ChunkStream::next()` checks `cut_points.is_empty() && offset != 0` THEN the system SHALL mark this validation branch as cold.
3. WHEN `decompress_auto_into()` performs format detection THEN the system SHALL order magic byte checks by frequency (Zstd first) and mark `UnknownFormat` as cold.
4. WHEN `ChunkStream::next()` handles `ErrorKind::Interrupted` THEN the system SHALL mark this path as unlikely.
5. WHEN checking `touches_end && !self.eof && len < self.max_size` THEN the system SHALL reorder conditions by likelihood and use `#[likely]`/`#[unlikely]` hints.
6. WHEN error paths construct messages THEN the system SHALL use static strings or pre-allocated error variants.

---

### HOT-004: Parallel Processing Threshold Tuning [P1]

**Location:** `chunking.rs:515-520`

**Description:** Fixed parallel threshold of 4 chunks may not be optimal.

**Acceptance Criteria:**
1. WHEN the parallel threshold is evaluated THEN the system SHALL consider both chunk count AND total data size.
2. WHEN the threshold is checked THEN the system SHALL make it configurable via environment variable `CHUNKER_PARALLEL_THRESHOLD`.
3. WHEN Rayon parallel iteration is used THEN the system SHALL consider chunk data size (e.g., total bytes > 64KB), not just count.
4. IF the average chunk size is less than 4-64KB THEN the system SHALL prefer sequential processing.
5. WHEN the number of available Rayon threads is 1 THEN the system SHALL bypass Rayon entirely.
6. WHEN processing chunks consistently smaller than 4KB THEN the system SHALL skip parallel hashing entirely.
7. WHEN `chunk_data_with_hash()` is called THEN the system SHALL add conditional parallel/sequential path based on chunk count.

---

### HOT-005: Hash Computation Optimization [P1]

**Location:** `chunking.rs:240-247, 488-495`, `hashing.rs:38-50`

**Description:** Per-chunk hasher allocation and match overhead.

**Acceptance Criteria:**
1. WHEN SHA-256 hashing is performed THEN the system SHALL verify SHA-NI instructions are used on x86_64 via assembly inspection.
2. WHEN BLAKE3 hashing is performed THEN the system SHALL ensure multi-threaded hashing for chunks larger than 64KB.
3. WHEN creating a hasher instance THEN the system SHALL evaluate using a thread-local hasher pool to avoid repeated initialization.
4. WHEN the `HashAlgorithm` enum is matched THEN the compiler SHALL generate a jump table, not cascading comparisons.
5. WHEN hashing chunks in parallel THEN the system SHALL batch small chunks (< 1-4KB) for sequential processing.
6. WHEN hashing chunk data THEN the system SHALL ensure data pointer is aligned to 16/32 byte boundaries when possible.
7. WHEN parallel hashing is performed THEN the system SHALL ensure each thread has its own hasher instance.

---

### HOT-006: Metrics Collection Overhead Reduction [P1]

**Location:** `chunking.rs:484-486, 545-547, 701-703, 742-745`

**Description:** Current 3 metrics calls per chunk adds overhead at scale.

**Acceptance Criteria:**
1. WHEN metrics are recorded per chunk THEN the overhead SHALL be less than 100 nanoseconds per chunk.
2. WHEN `counter!()` and `histogram!()` macros are invoked THEN the system SHALL use cached counter handles rather than macro-based lookups.
3. WHEN recording metrics in hot paths THEN the system SHALL implement batched metrics aggregation using thread-local counters that flush periodically.
4. IF metrics feature is disabled at compile time THEN the system SHALL compile out all metrics code with zero runtime cost.
5. WHEN metrics are enabled THEN the system SHALL use thread-local aggregation to avoid atomic contention.
6. IF environment variable `CHUNKER_METRICS_SAMPLE_RATE` is set THEN the system SHALL sample metrics at the specified rate.
7. WHEN histogram metrics are recorded THEN the system SHALL use pre-computed bucket boundaries.
8. IF chunk throughput exceeds 1M chunks/second THEN the system SHALL automatically switch to sampled metrics.

---

### HOT-007: Eliminate Dynamic Dispatch in Hot Paths [P1]

**Location:** `compression.rs:264-343`

**Description:** `AutoDecompressReader` uses `Box<dyn Read>` causing virtual table lookups.

**Acceptance Criteria:**
1. WHEN `AutoDecompressReader` is used THEN the system SHALL provide an enum-based implementation for static dispatch.
2. WHEN `HashAlgorithm` enum is matched THEN the system SHALL use const generics or specialized functions for monomorphization.
3. WHEN `ChunkStream<R: Read>` is generic THEN the system SHALL ensure Read trait calls are monomorphized.
4. WHEN `CompressionStrategy` is matched in `compress()` THEN the system SHALL inline the match or use const generics.
5. IF a hot path function takes a trait object THEN the system SHALL provide a generic `impl Trait` alternative.

---

## 3. Memory & Allocation

### MEM-001: Eliminate Heap Allocations in Hot Paths [P0]

**Location:** `chunking.rs:420, 394, 253, 294-312`

**Description:** Several allocations occur in per-chunk processing.

**Acceptance Criteria:**
1. WHEN `cut_points` Vec is created THEN the system SHALL use stack-allocated `SmallVec` or `ArrayVec` with capacity 8-16.
2. WHEN `pending_chunks: VecDeque` is initialized THEN the system SHALL pre-allocate with `VecDeque::with_capacity()`.
3. WHEN `process_chunk` constructs `ChunkMetadata` THEN `Bytes::slice()` SHALL perform zero-copy reference counting.
4. WHEN `effective_read_slice_cap()` or `effective_async_buffer_limit()` is called THEN the system SHALL cache the result in a `OnceLock<usize>` static.
5. WHEN error paths construct messages THEN the system SHALL avoid heap allocation.
6. WHEN `Vec::with_capacity()` is used THEN the system SHALL use accurate capacity estimates.

---

### MEM-002: Buffer Pooling for Compression [P1]

**Location:** `compression.rs:26-175`

**Description:** Compression buffers should be pooled and reused.

**Acceptance Criteria:**
1. WHEN `CompressionScratch` is used THEN the system SHALL maintain buffer capacity across operations.
2. WHEN thread-local buffer pools are requested THEN the system SHALL provide `ThreadLocal<CompressionScratch>`.
3. WHEN buffer pools are used THEN the system SHALL limit maximum pool size.
4. WHEN compression operations complete THEN the system SHALL NOT shrink buffers below minimum capacity.
5. WHEN using `CompressionScratch` THEN the system SHALL verify zero allocations after warmup.
6. WHEN reusing buffers THEN `clear()` SHALL NOT deallocate memory.

---

### MEM-003: Buffer Management Optimization [P1]

**Location:** `chunking.rs:577-582, 457, 542, 699, 741`

**Description:** Buffer operations should be optimized for chunking patterns.

**Acceptance Criteria:**
1. WHEN `BytesMut::reserve()` and `resize()` are called THEN the system SHALL use exponential growth (2x) with configurable cap.
2. WHEN `buffer.split_to()` is called THEN the system SHALL verify O(1) operation without hidden copies.
3. WHEN `self.buffer.truncate(start)` is called THEN the system SHALL verify no memory deallocation.
4. WHEN `BytesMut::freeze()` creates `Bytes` THEN allocation SHALL be shared via atomic ref counting.
5. WHEN `buffer.resize(start + read_size, 0)` is called THEN the system SHALL evaluate `set_len()` when buffer will be overwritten.
6. WHEN the buffer grows beyond `max_size * 2` THEN the system SHALL implement buffer compaction.

---

### MEM-004: Zero-Copy Data Paths [P1]

**Location:** `chunking.rs:481, 253`

**Description:** Minimize data copying throughout the pipeline.

**Acceptance Criteria:**
1. WHEN `ChunkMetadata` payload is created THEN the system SHALL use `Bytes::slice` for zero-copy views.
2. WHEN chunks are passed to compression THEN the system SHALL accept borrowed slices.
3. WHEN NIF bindings receive Elixir binaries THEN the system SHALL process without copying where possible.
4. WHEN streaming from `AsyncRead` THEN the system SHALL minimize intermediate buffer copies.
5. WHEN returning results THEN the system SHALL provide options for borrowed vs owned data.

---

### MEM-005: Cache-Friendly Data Structures [P2]

**Location:** `chunking.rs:62-75, 270-280`

**Description:** Data structure layouts should optimize cache efficiency.

**Acceptance Criteria:**
1. WHEN `ChunkMetadata` is stored in vectors THEN layout SHALL be optimized for sequential access.
2. WHEN `ChunkMetadata` fields are ordered THEN hot fields (hash, offset, length) SHALL be contiguous.
3. WHEN processing chunks in parallel THEN the system SHALL avoid false sharing via padding/alignment.
4. WHEN `ChunkStream` is instantiated THEN the system SHALL evaluate `#[repr(align(64))]` for cache line alignment.
5. WHEN buffer sizes are chosen THEN the system SHALL align to cache line boundaries (64 bytes).
6. IF Structure-of-Arrays provides benefit THEN the system SHALL offer SoA alternatives.
7. WHEN gear table is accessed THEN it SHALL be cache-line aligned (64-byte).

---

### MEM-006: Memory Limits and Budgets [P2]

**Location:** Various

**Description:** Configurable memory limits for containerized deployments.

**Acceptance Criteria:**
1. WHEN `MemoryBudget` is configured THEN the system SHALL track total allocations.
2. WHEN memory usage approaches `soft_limit_bytes` THEN the system SHALL proactively evict cache entries.
3. WHEN memory usage exceeds `hard_limit_bytes` THEN the system SHALL return `Error::MemoryLimitExceeded`.
4. WHEN streaming large files THEN the system SHALL enforce `max_buffer_size` (current: 256MB ceiling).
5. WHEN buffer allocation fails THEN the system SHALL return `ChunkingError::BufferLimitExceeded` instead of panicking.

---

## 4. Parallelism & Concurrency

### PAR-001: Rayon Thread Pool Optimization [P1]

**Location:** `chunking.rs:223-256, 515-520`

**Description:** Rayon configuration needs tuning for chunking workloads.

**Acceptance Criteria:**
1. WHEN the library initializes THEN the system SHALL respect `RAYON_NUM_THREADS` environment variable.
2. WHEN no variable is set THEN the system SHALL default to `num_cpus::get()` threads.
3. WHEN configuring the thread pool THEN the system SHALL provide `ChunkerConfig::with_thread_pool(rayon::ThreadPool)`.
4. WHEN processing chunks in parallel THEN the system SHALL use work-stealing scheduling.
5. WHEN chunks have highly variable sizes THEN the system SHALL consider chunk size when scheduling.
6. WHEN used in multi-tenant applications THEN the system SHALL support isolated thread pools per instance.
7. WHEN the Rayon pool is saturated THEN new tasks SHALL queue with bounded depth.

---

### PAR-002: Lock-Free Metrics Collection [P1]

**Location:** `chunking.rs:282, 484-486`

**Description:** Atomic counters cause contention in parallel contexts.

**Acceptance Criteria:**
1. WHEN metrics are enabled THEN the system SHALL use atomic counters for basic metrics.
2. WHEN updating histograms in hot paths THEN the system SHALL use lock-free implementations.
3. IF metrics would require a lock THEN the system SHALL use thread-local aggregation with periodic merging.
4. WHEN `TRACE_SAMPLE_COUNTER` is incremented THEN the system SHALL use thread-local counters with aggregation.
5. WHEN multiple `ChunkStream` instances operate concurrently THEN they SHALL share no mutable static state.
6. WHEN metrics counters are incremented THEN the system SHALL flush to global counters at batch boundaries.

---

### PAR-003: Pipeline Parallelism Architecture [P2]

**Location:** N/A (new architecture)

**Description:** Enable concurrent chunking, hashing, and compression stages.

**Acceptance Criteria:**
1. WHEN streaming THEN the system SHALL support concurrent execution of chunking, hashing, and compression in pipeline.
2. WHEN pipeline parallelism is enabled THEN chunk N+1 MAY begin FastCDC while chunk N is hashed AND chunk N-1 is compressed.
3. WHEN pipeline stages have different throughput THEN the system SHALL implement bounded channels with configurable capacity.
4. IF downstream stage is slower THEN the system SHALL apply backpressure.
5. WHEN configuring pipeline THEN the system SHALL provide `PipelineBuilder` API.
6. WHEN disabled THEN the system SHALL fall back to current model for backward compatibility.

---

### PAR-004: Cooperative Scheduling for NIF [P2]

**Location:** NIF bindings

**Description:** Long-running operations should yield to BEAM scheduler.

**Acceptance Criteria:**
1. WHEN running as NIF THEN long operations SHALL yield after processing `yield_chunk_count` chunks.
2. WHEN streaming via iterator THEN each `next()` SHALL complete within `max_iteration_time_us` (default 1000us).
3. WHEN batch operations detect scheduler pressure THEN they SHALL reduce parallelism dynamically.
4. WHEN async chunking is used THEN the system SHALL use `tokio::task::yield_now()` between emissions.
5. IF cooperative mode is disabled THEN the system SHALL skip yield points for maximum throughput.

---

## 5. Algorithm Optimizations

### ALG-001: SIMD Base32 Encoding [P2]

**Location:** `hashing.rs:58-101`

**Description:** Base32 encoding/decoding can benefit from SIMD.

**Acceptance Criteria:**
1. WHEN encoding to Nix base32 THEN the system SHALL support SIMD acceleration on x86_64 (AVX2) and aarch64 (NEON).
2. WHEN SIMD is not available THEN the system SHALL fall back to scalar implementation transparently.
3. WHEN SIMD base32 is used THEN throughput SHALL be at least 2x scalar for 32-byte inputs.
4. WHEN decoding THEN the system SHALL use lookup table optimization with `NIX_BASE32_INVERSE`.
5. WHEN encoding batch operations THEN the system SHALL process multiple hashes in parallel.

---

### ALG-002: FastCDC Gear Hash SIMD [P2]

**Location:** FastCDC crate (external)

**Description:** Gear table lookups could use SIMD gather instructions.

**Acceptance Criteria:**
1. WHEN FastCDC processes data THEN gear table lookup SHALL be cache-friendly (sequential access).
2. WHEN gear table is initialized THEN it SHALL be computed once and cached statically.
3. IF custom gear tables are supported THEN the system SHALL allow NAR-optimized tables.
4. WHEN gear hash lookup occurs THEN the system SHALL investigate SIMD gather for parallel lookups (4-8 bytes).
5. WHEN the compiler can unroll the inner loop THEN operations SHALL be constant-time without data-dependent branches.

---

### ALG-003: Compression Dictionary Preloading [P2]

**Location:** `compression.rs`

**Description:** Pre-trained dictionaries improve compression for NAR data.

**Acceptance Criteria:**
1. WHEN zstd compression is used THEN the system SHALL support preloaded `EncoderDictionary`.
2. WHEN dictionaries are used THEN they SHALL be trained on representative NAR samples.
3. WHEN dictionary compression is enabled THEN the system SHALL achieve at least 10% better compression.
4. WHEN decompressing with dictionaries THEN the system SHALL support preloaded `DecoderDictionary`.
5. IF dictionary is not available THEN the system SHALL fall back transparently.

---

### ALG-004: Adaptive Compression Strategies [P2]

**Location:** `compression.rs:189-194`

**Description:** Compression should adapt to content characteristics.

**Acceptance Criteria:**
1. WHEN `CompressionStrategy::Adaptive` is selected THEN the system SHALL sample content and choose algorithm.
2. WHEN content is already compressed (detected via magic bytes) THEN the system SHALL skip recompression.
3. WHEN compression ratio falls below `min_ratio_threshold` (0.95) THEN the system SHALL store uncompressed.
4. IF compression time exceeds `max_compression_time_ms` THEN the system SHALL fall back to faster algorithm.
5. WHEN detecting content type THEN the system SHALL analyze first 4KB: binary, text, compressed, multimedia.

---

### ALG-005: Compression Encoder Pooling [P2]

**Location:** `compression.rs:537-561`

**Description:** Zstd encoder creation overhead can be amortized.

**Acceptance Criteria:**
1. WHEN `compress_zstd_into_internal()` creates encoder THEN the system SHALL use thread-local encoder pool.
2. WHEN magic byte detection occurs THEN the system SHALL use branchless comparison or lookup table.
3. WHEN LZ4 compression is performed THEN the system SHALL use frame encoder with pre-sized buffers.
4. WHEN decompression readers use `.take(MAX)` THEN the system SHALL verify no measurable hot path overhead.

---

## 6. I/O & Streaming

### IO-001: Adaptive Buffer Sizing [P1]

**Location:** `chunking.rs:577-582`

**Description:** Buffer sizes should adapt to workload characteristics.

**Acceptance Criteria:**
1. WHEN initializing `ChunkStream` THEN the system SHALL start with conservative allocation (min of `min_chunk_size` and 64KB).
2. WHEN buffer growth is required THEN the system SHALL use exponential growth (2x) up to `CHUNKER_READ_SLICE_CAP_BYTES` (default 16MB).
3. WHEN processing chunks THEN the system SHALL track rolling average chunk size and pre-allocate based on patterns.
4. IF workload exhibits consistent chunk sizes THEN the system SHALL stabilize buffer allocation.
5. WHEN memory pressure is detected THEN the system SHALL reduce buffer sizes gracefully.
6. WHEN configuring buffer behavior THEN the system SHALL provide `BufferStrategy::Fixed`, `Adaptive`, and `PreAllocated`.

---

### IO-002: Memory-Mapped File Support [P2]

**Location:** N/A (new feature)

**Description:** Enable mmap for large file processing.

**Acceptance Criteria:**
1. WHEN a file path is provided THEN the system SHALL offer `ChunkStream::open_mmap(path)`.
2. WHEN using mmap THEN the system SHALL pass mapped memory directly to FastCDC without copies.
3. IF mmap fails THEN the system SHALL fall back to buffered I/O with warning.
4. WHEN mmap is used THEN the system SHALL call `madvise(MADV_SEQUENTIAL)`.
5. WHEN chunker completes/drops THEN the system SHALL properly unmap.
6. WHEN mmap is used with very large files (>1GB) THEN the system SHALL support windowed/sliding mmap.

---

### IO-003: I/O Read Buffer Optimization [P2]

**Location:** `chunking.rs:584`

**Description:** Read operations should align with filesystem characteristics.

**Acceptance Criteria:**
1. WHEN reading from files THEN the system SHALL use read sizes that are multiples of filesystem block size (4KB).
2. WHEN the `io_uring` feature is enabled (Linux 5.1+) THEN the system SHALL use async I/O.
3. WHEN configuring read behavior THEN the system SHALL provide `CHUNKER_READ_SLICE_CAP_BYTES` and `CHUNKER_READAHEAD_BYTES`.
4. IF the data source supports vectored I/O THEN the system SHALL use `readv()`.
5. WHEN using direct I/O (`O_DIRECT`) THEN the system SHALL ensure page-aligned buffers.
6. WHEN reading THEN the system SHALL align sizes to page boundaries (4KB).

---

### IO-004: Async Stream Optimization [P2]

**Location:** `chunking.rs` async paths

**Description:** Async streaming should integrate efficiently with runtimes.

**Acceptance Criteria:**
1. WHEN using `ChunkStreamAsync` THEN the system SHALL avoid blocking the async runtime.
2. WHEN `async-stream` feature is enabled THEN the system SHALL be compatible with Tokio, async-std, and smol.
3. IF async reader yields `Poll::Pending` THEN the system SHALL properly suspend and resume without busy-waiting.
4. WHEN async streaming is used THEN the system SHALL provide `poll_next_chunk()` for manual polling.
5. WHEN async buffers are sized THEN the system SHALL use adaptive sizing based on observed throughput.
6. IF `spawn_blocking` is available THEN the system SHALL offload CPU-intensive hashing.

---

## 7. NIF Performance

### NIF-001: NIF Binary Handling Efficiency [P1]

**Location:** NIF bindings

**Description:** Minimize copying between Elixir and Rust.

**Acceptance Criteria:**
1. WHEN receiving Elixir binaries THEN the system SHALL use `Binary<'a>` lifetime-bound references.
2. WHEN NIF needs to modify binary THEN the system SHALL copy to `OwnedBinary` only when required.
3. WHEN returning chunk metadata THEN the system SHALL return tuples without payload unless requested.
4. IF caller needs payloads THEN the system SHALL provide `chunk_data_with_payloads/4` using sub-binary refs.
5. WHEN processing streaming data THEN the system SHALL support resource-based API:
   - `chunker_open(options) -> resource`
   - `chunker_feed(resource, binary) -> {:ok, [chunks]} | {:more}`
   - `chunker_finish(resource) -> {:ok, [final_chunks]}`

---

### NIF-002: NIF Batch APIs [P2]

**Location:** NIF bindings

**Description:** Batch operations amortize NIF call overhead.

**Acceptance Criteria:**
1. WHEN processing multiple chunks THEN the system SHALL provide batch NIF APIs:
   - `chunk_data_batch(list_of_binaries)`
   - `compress_batch(list_of_binaries, strategy)`
2. WHEN batch size exceeds `parallel_threshold` THEN the system SHALL use parallel execution.
3. WHEN batch processing fails for single item THEN the system SHALL return partial success results.
4. WHEN batch sizes are very large THEN the system SHALL process in chunks to maintain progress visibility.
5. WHEN returning results THEN the system SHALL use `OwnedBinary` with pre-allocated capacity.

---

### NIF-003: DirtyCpu Scheduler Optimization [P2]

**Location:** NIF bindings

**Description:** Long-running operations should use Dirty CPU schedulers.

**Acceptance Criteria:**
1. WHEN NIF performs CPU-intensive work THEN it SHALL be annotated with `#[rustler::nif(schedule = "DirtyCpu")]`.
2. IF NIF operation takes longer than 1ms THEN the system SHALL yield periodically or split work.
3. WHEN errors occur THEN the system SHALL return Erlang atoms rather than string messages.
4. WHEN data is passed to NIFs THEN the system SHALL use Rustler's `Binary` for zero-copy access.

---

## 8. Observability

### OBS-001: Zero-Cost Tracing [P1]

**Location:** `chunking.rs:499-505, 717-721`

**Description:** Tracing must have zero cost when disabled.

**Acceptance Criteria:**
1. WHEN tracing is disabled at compile time THEN all macros SHALL compile to no-ops with zero cost.
2. WHEN tracing is enabled but level is above event level THEN level checking SHALL occur before argument evaluation.
3. WHEN sampling strategy is used THEN the system SHALL use atomic counter increments without locks.
4. WHEN configuring runtime levels THEN the system SHALL support dynamic changes via `tracing_subscriber::reload`.
5. IF tracing is disabled at compile time THEN all code SHALL be eliminated via `#[cfg]`.
6. WHEN `tracing::enabled!` check occurs THEN it SHALL compile to single static boolean load.
7. WHEN `instrument` attributes are used THEN they SHALL be on cold/setup paths only.

---

### OBS-002: Performance Counters [P2]

**Location:** Metrics system

**Description:** Real-time performance counters for production monitoring.

**Acceptance Criteria:**
1. WHEN `metrics` feature is enabled THEN the system SHALL emit counters via `metrics` crate facade.
2. WHEN chunks are emitted THEN the system SHALL increment: `chunker.chunks_emitted_total`, `chunker.bytes_processed_total`.
3. WHEN compression completes THEN the system SHALL record: `compression_bytes_in`, `compression_bytes_out`, `compression_ratio`.
4. WHEN hashing completes THEN the system SHALL record: `hash_operations_total`, `hash_bytes_total`.
5. WHEN cache operations occur THEN the system SHALL record: `cache_hits_total`, `cache_misses_total`.
6. WHEN errors occur THEN the system SHALL increment: `errors_total{type="io|bounds|compression"}`.

---

### OBS-003: Latency Histograms [P2]

**Location:** Metrics system

**Description:** Latency distributions for tail latency analysis.

**Acceptance Criteria:**
1. WHEN chunk processing completes THEN the system SHALL record: `chunker.chunk_latency_seconds` histogram.
2. WHEN compression completes THEN the system SHALL record: `compression_latency_seconds` by algorithm.
3. WHEN hashing completes THEN the system SHALL record: `hash_latency_seconds` by algorithm.
4. WHEN histograms are configured THEN the system SHALL support custom bucket boundaries.
5. IF p99 latency exceeds threshold THEN the system SHALL emit warning log.

---

### OBS-004: Bottleneck Identification Tools [P3]

**Location:** Profiling infrastructure

**Description:** Built-in profiling for performance debugging.

**Acceptance Criteria:**
1. WHEN `CHUNKER_PROFILE=1` is set THEN the system SHALL emit detailed timing spans via `tracing`.
2. WHEN profiling is enabled THEN the system SHALL report time in: FastCDC, hashing, compression, I/O wait.
3. WHEN `flamegraph` feature is enabled THEN the system SHALL support `tracing-flame` integration.
4. WHEN bottleneck analysis is requested THEN the system SHALL identify slowest phase as percentage.
5. IF I/O wait exceeds 50% of total time THEN the system SHALL suggest increasing buffer sizes.

---

## 9. Testing & Benchmarking

### TEST-001: Comprehensive Benchmark Suite [P1]

**Location:** `benches/`

**Description:** Benchmarks covering all critical code paths.

**Acceptance Criteria:**
1. WHEN running benchmarks THEN the system SHALL measure throughput for:
   - FastCDC raw chunking
   - `chunk_data` (eager)
   - `ChunkStream` (streaming)
   - SHA256 and BLAKE3 hashing
   - Compression: zstd (levels 1,3,9,19), lz4, xz, bzip2
   - Ed25519 signing/verification
   - Nix base32 encode/decode
2. WHEN running benchmarks THEN the system SHALL test sizes: 1KB, 64KB, 1MB, 10MB, 100MB, 1GB.
3. WHEN running benchmarks THEN the system SHALL test patterns: zeros, random, realistic NAR content.
4. WHEN benchmarks complete THEN the system SHALL output JSON for CI integration.
5. WHEN running benchmarks THEN the system SHALL include memory allocation profiling.

---

### TEST-002: Performance Regression Detection [P1]

**Location:** CI pipeline

**Description:** Automated regression detection in CI.

**Acceptance Criteria:**
1. WHEN PR is submitted THEN CI SHALL run benchmark suite against main branch baseline.
2. WHEN results deviate by more than 5% THEN the system SHALL flag as potential regression.
3. WHEN regression is detected THEN the system SHALL report specific benchmarks and percentages.
4. WHEN benchmarks complete THEN the system SHALL store results for trend analysis.
5. WHEN running detection THEN the system SHALL use statistical significance testing.
6. IF benchmark variance exceeds 10% THEN the system SHALL exclude from regression gates.
7. WHEN regressions are reported THEN CI SHALL include flame graphs.

---

### TEST-003: Edge Case Testing [P2]

**Location:** Test suite

**Description:** Tests for boundary conditions and edge cases.

**Acceptance Criteria:**
1. WHEN `chunk_data` receives empty input THEN the system SHALL return empty `Vec<ChunkMetadata>`.
2. WHEN `ChunkStream` receives EOF immediately THEN the system SHALL iterate zero times without error.
3. WHEN compression receives empty input THEN the system SHALL return valid compressed output.
4. WHEN a chunk exactly equals `max_size` THEN the system SHALL emit correctly.
5. WHEN input size approaches `usize::MAX` THEN the system SHALL handle overflow gracefully.
6. WHEN decompression would exceed `MAX_DECOMPRESSED_SIZE` THEN all codecs SHALL return `SizeExceeded`.

---

### TEST-004: Miri Validation [P2]

**Location:** CI pipeline

**Description:** Memory safety validation via Miri.

**Acceptance Criteria:**
1. WHEN CI runs THEN Miri SHALL execute test suite to detect: use-after-free, out-of-bounds, invalid pointers, data races.
2. IF Miri detects undefined behavior THEN CI SHALL fail with detailed diagnostics.
3. WHEN Miri is too slow THEN a representative subset SHALL run in CI with full suite nightly.
4. WHEN new unsafe code is added THEN corresponding Miri tests SHALL be required.
5. WHEN unsafe code exists THEN each block SHALL have `// SAFETY:` comment.

---

### TEST-005: Test Data Diversity [P2]

**Location:** Test fixtures

**Description:** Realistic test data for accurate benchmarking.

**Acceptance Criteria:**
1. WHEN benchmarking THEN test data SHALL include:
   - Highly compressible (zeros, patterns)
   - Incompressible (random, encrypted)
   - Realistic NAR content (ELF, libraries, text)
   - Edge cases (tiny, huge, pathological)
2. WHEN using NAR fixtures THEN they SHALL be generated from real Nix store paths.
3. WHEN results are reported THEN data characteristics SHALL be included.
4. WHEN comparing codecs THEN same data sets SHALL be used.

---

## 10. API Consistency

### API-001: Missing `_into()` Variants [P1]

**Location:** `compression.rs`

**Description:** Bzip2 and XZ lack buffer-reusing variants.

**Acceptance Criteria:**
1. WHEN using bzip2 compression THEN the system SHALL provide `compress_bzip2_into(data, output)`.
2. WHEN using XZ compression THEN the system SHALL provide `compress_xz_into(data, output)`.
3. WHEN `_into()` variants are called THEN the output buffer SHALL be cleared before writing.
4. WHEN using `CompressionScratch` THEN all strategies SHALL use `_into()` internally.
5. WHEN `_into()` variants are used THEN the system SHALL return bytes written or provide via `output.len()`.

---

### API-002: Consistent Error Handling [P2]

**Location:** All modules

**Description:** Unified error types across modules.

**Acceptance Criteria:**
1. WHEN errors occur THEN the system SHALL return typed errors: `ChunkingError`, `CompressionError`, `HashingError`, `SigningError`.
2. WHEN errors contain context THEN the system SHALL use structured fields.
3. WHEN errors are displayed THEN the system SHALL produce machine-parseable lowercase_snake_case format.
4. WHEN NIF functions error THEN the system SHALL return Erlang atoms.
5. WHEN errors need conversion THEN the system SHALL provide `From` implementations.
6. WHEN using `thiserror` THEN all types SHALL implement `std::error::Error` and `Display`.

---

### API-003: Batch Processing APIs [P2]

**Location:** Public API

**Description:** Batch operations for high-throughput use cases.

**Acceptance Criteria:**
1. WHEN processing multiple chunks for hashing THEN the system SHALL provide `hash_batch(chunks: &[&[u8]])`.
2. WHEN processing multiple items for compression THEN the system SHALL provide `compress_batch(items, strategy)`.
3. WHEN using batch APIs THEN the system SHALL parallelize when batch size exceeds threshold.
4. WHEN batch processing fails for single item THEN the system SHALL return `Result<Vec<Result<T, E>>, E>`.
5. WHEN batch processing files THEN the system SHALL provide `batch_chunk_files(paths, options)`.
6. WHEN processing >100 files THEN the system SHALL report progress metrics.

---

## 11. Platform & Build Optimizations

### PLT-001: SIMD CPU Feature Detection [P1]

**Location:** Runtime detection

**Description:** Automatic selection of optimal code paths.

**Acceptance Criteria:**
1. WHEN SIMD operations are available THEN the system SHALL detect AVX2, AVX-512, NEON, SHA-NI at runtime.
2. WHEN features are detected THEN the system SHALL cache result for process lifetime.
3. WHEN multiple implementations exist THEN the system SHALL dispatch to fastest available.
4. WHEN cross-compiling THEN the system SHALL support `target_feature` for static selection.
5. WHEN feature detection fails THEN the system SHALL fall back to portable scalar.
6. WHEN running diagnostics THEN the system SHALL expose detected features via public API.

---

### PLT-002: Compile-Time Optimization [P1]

**Location:** `Cargo.toml`, build configuration

**Description:** Ensure maximum compiler optimizations.

**Acceptance Criteria:**
1. WHEN building release THEN the system SHALL use LTO with `lto = true` (current).
2. WHEN building release THEN the system SHALL use `codegen-units = 1` (current).
3. WHEN `asm` feature is enabled THEN the system SHALL use assembly-optimized SHA256.
4. WHEN targeting specific architectures THEN the system SHALL support `RUSTFLAGS="-C target-cpu=native"`.
5. WHEN building production THEN the system SHALL strip debug symbols (current).
6. WHEN PGO is available THEN the build system SHALL document the process.

---

### PLT-003: Profile-Guided Optimization [P2]

**Location:** Build process

**Description:** PGO for maximum performance.

**Acceptance Criteria:**
1. WHEN PGO is requested THEN the build system SHALL provide documented workflow.
2. WHEN generating profiles THEN the system SHALL include representative benchmarks.
3. WHEN PGO is enabled THEN the system SHALL verify improvements via automated benchmarks.
4. IF profiles are incompatible THEN the build SHALL fail with clear error.
5. WHEN distributing pre-built binaries THEN the system SHALL document PGO usage.

---

### PLT-004: Feature Flag Optimization [P2]

**Location:** `Cargo.toml`

**Description:** Fine-grained feature control.

**Acceptance Criteria:**
1. WHEN building THEN users SHALL independently enable/disable: each codec, each hash, signing, async, NIF, telemetry.
2. WHEN building with minimal features THEN binary size SHALL be documented.
3. WHEN a feature is disabled THEN its dependencies SHALL not be compiled.
4. WHEN default features are selected THEN they SHALL represent common use case.
5. WHEN features are combined THEN the system SHALL validate at compile time.

---

## 12. NAR-Specific Optimizations

### NAR-001: NAR File Structure Analysis [P3]

**Location:** N/A (new feature)

**Description:** NAR-aware chunking for better deduplication.

**Acceptance Criteria:**
1. WHEN processing NAR THEN the system SHALL parse magic bytes to identify format.
2. WHEN NAR header is detected THEN the system SHALL provide option to emit header as separate chunk.
3. WHEN chunking NAR THEN the system SHALL expose `NarAwareChunker` mode biasing boundaries toward entry boundaries.
4. IF NAR-aware mode is enabled AND entry boundary is within `avg_size / 4` of natural cut THEN the system SHALL prefer entry boundary.
5. WHEN NAR-aware chunking is used THEN the system SHALL maintain metadata mapping chunks to entries.
6. WHEN processing large NAR (>100MB) THEN the system SHALL support streaming parsing.

---

### NAR-002: NAR-Optimized Chunk Sizes [P3]

**Location:** Configuration

**Description:** Default sizes optimized for NAR patterns.

**Acceptance Criteria:**
1. WHEN configuring chunker THEN the system SHALL provide NAR-optimized defaults: min=128KB, avg=512KB, max=2MB.
2. WHEN chunk contains only padding bytes THEN the system SHALL NOT emit standalone padding chunks.
3. WHEN processing executables within NAR THEN the system SHALL support detecting ELF/Mach-O boundaries.
4. WHEN same file content appears in multiple NARs THEN chunk boundaries SHALL be deterministic.

---

### NAR-003: Chunk Hash Caching [P3]

**Location:** N/A (new feature)

**Description:** Cache chunk hashes to avoid redundant computation.

**Acceptance Criteria:**
1. WHEN `HashCache` is enabled THEN the system SHALL maintain LRU cache mapping fingerprints to hashes.
2. WHEN fingerprint matches THEN the system SHALL verify via content comparison.
3. WHEN cache reaches `max_entries` THEN the system SHALL evict LRU entries.
4. WHEN cache reaches `max_memory_bytes` THEN the system SHALL evict until under budget.
5. IF cache statistics are requested THEN the system SHALL report: hit_count, miss_count, eviction_count.
6. WHEN multi-threaded THEN the cache SHALL use concurrent data structure (DashMap).

---

## Non-Functional Requirements

### NFR-1: Performance Targets

| Metric | Target |
|--------|--------|
| FastCDC raw throughput | >= 2.5 GB/s (x86_64 AVX2) |
| BLAKE3 hashing | >= 3-5 GB/s (multi-core SIMD) |
| SHA256 hashing (SHA-NI) | >= 1 GB/s |
| Zstd level 3 compression | >= 400-500 MB/s |
| LZ4 compression | >= 2 GB/s |
| End-to-end pipeline | >= 300 MB/s |
| ChunkStream::next() overhead | < 100ns per call |
| Per-chunk latency (< 1MB) | < 1 microsecond (excl. I/O) |
| p99 latency for 1MB chunk | < 5-10ms |
| 100M chunks processing time | < 10 minutes (8-core, 32GB) |

### NFR-2: Memory Efficiency

1. Hot path SHALL perform zero heap allocations for common case (batch <= 16 chunks).
2. `ChunkStream` initial footprint SHALL be < 128KB before data read.
3. Metrics overhead per thread SHALL be < 4KB.
4. When processing 100MB files, peak memory SHALL not exceed 50MB (excluding input).
5. When streaming large files, memory SHALL use no more than 2x buffer limit.

### NFR-3: Backward Compatibility

1. Public API signatures SHALL remain unchanged.
2. New APIs SHALL be additive.
3. Default behaviors that change SHALL be configurable to restore previous behavior.
4. Serialization formats (serde) SHALL remain compatible.

### NFR-4: Build Impact

1. Incremental build time increase SHALL be no more than 20%.
2. SIMD code SHALL be feature-gated.
3. Release binary size increase SHALL be no more than 10%.

### NFR-5: Platform Support

1. x86_64 (AVX2) and aarch64 (NEON) SHALL be supported.
2. Fallback implementations SHALL exist for other platforms.
3. Cross-compilation SHALL work correctly.
4. Rust 2024 edition, MSRV 1.85+ SHALL be maintained.
5. NIF SHALL support Erlang/OTP 24+, Elixir 1.14+.

### NFR-6: Security

1. Cryptographic keys SHALL be zeroized after use.
2. Decompression size limits SHALL be enforced.
3. All unsafe blocks SHALL have safety documentation.

### NFR-7: Reliability

1. Fuzz testing SHALL handle arbitrary input without panics for 24+ hours.
2. Malformed data SHALL return errors, not crash.
3. Decompression bombs SHALL be detected within 100ms.
4. Test coverage SHALL exceed 90%.

---

## Appendix A: Hot Path Code Locations Summary

| Hot Path | File | Lines | Issue | Priority |
|----------|------|-------|-------|----------|
| Bzip2 bomb check | compression.rs | 714 | Uses `==` instead of `>=` | P0 |
| Integer overflow | chunking.rs | 214-216 | Unchecked u32 casts | P0 |
| FastCDC construction | chunking.rs | 412-417 | Recreated per next() | P0 |
| SHA256 hashing | chunking.rs | 242-244, 489-492 | Per-chunk hasher alloc | P1 |
| Metrics recording | chunking.rs | 484-486 | 3x metrics per chunk | P1 |
| Parallel threshold | chunking.rs | 515 | Fixed at 4 chunks | P1 |
| Buffer resize | chunking.rs | 582 | Zero-initialization | P1 |
| Tracing check | chunking.rs | 499-505 | Sampled but overhead | P1 |
| Dynamic dispatch | compression.rs | 264-343 | Box<dyn Read> | P1 |

---

## Appendix B: Priority Legend

- **P0**: Critical - Must fix before any release. Security issues, data corruption, crashes.
- **P1**: High - Major performance impact. Should be addressed in next sprint.
- **P2**: Medium - Significant improvement. Plan for upcoming releases.
- **P3**: Low - Nice to have. Future enhancement consideration.

---

## Appendix C: Source Document Consolidation

This document consolidates requirements from 6 source documents:

1. `requirements_v1.md` - Performance profiling, hot path, memory, SIMD, compression (24 requirements)
2. `requirements_v2.md` - Streaming, concurrency, I/O, NIF, compile-time, observability (18 requirements)
3. `requirements_v3.md` - Code quality, testing, edge cases, documentation, platform-specific (26 requirements)
4. `requirements_v4.md` - NAR-specific, caching, batch processing, resource management (25 requirements)
5. `requirements_v5.md` - Hot path micro-optimizations, allocations, cache efficiency (12 requirements)
6. `requirements_v6.md` - FastCDC loop, hash computation, SIMD, branch prediction (12 requirements)

**Total original requirements:** ~117 across 6 documents

**Consolidated requirements:** 47 unique requirements in 12 categories

**Merge notes:**
- Bug fixes (BUG-001, BUG-002) identified in v1, v3
- Hot path optimizations merged from v1, v5, v6 (significant overlap)
- Memory requirements merged from v1, v4, v5
- Parallelism requirements merged from v1, v2, v4
- Testing requirements merged from v1, v3, v4
- NAR-specific requirements from v4 preserved as lower priority
- Metrics/observability merged from v1, v2, v4, v5

The source documents can be deleted after this consolidation is approved.
