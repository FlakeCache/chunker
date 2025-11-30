# Requirements Document: Chunker Library Improvements

## Introduction

This document specifies requirements for bug fixes, feature additions, and optimizations identified in the Chunker library (v0.1.0-beta6). The improvements address security issues, API consistency gaps, and code quality concerns.

---

## Requirements

### Requirement 1: Fix Bzip2 Decompression Bomb Detection

**User Story:** As a security-conscious developer, I want the bzip2 decompression to correctly detect decompression bombs, so that malicious payloads cannot exhaust memory.

#### Acceptance Criteria

1. WHEN `decompress_bzip2` or `decompress_bzip2_into` processes data THEN it SHALL use `>` comparison (not `==`) for size limit checking, consistent with zstd and lz4 implementations.
2. WHEN decompressed output exceeds `MAX_DECOMPRESSED_SIZE` (1 GB) THEN it SHALL return `CompressionError::SizeExceeded`.
3. WHEN the fix is applied THEN existing tests SHALL continue to pass.
4. WHEN the fix is applied THEN a new test SHALL verify bomb detection works for payloads that decompress to just over the limit.

**Location:** `src/compression.rs:714`

---

### Requirement 2: Add `compress_bzip2_into()` Function

**User Story:** As a developer using buffer reuse patterns, I want a `compress_bzip2_into()` function, so that I can compress bzip2 data without allocating new buffers each time.

#### Acceptance Criteria

1. WHEN `compress_bzip2_into(data, level, output)` is called THEN it SHALL compress data into the provided output buffer.
2. WHEN the function is implemented THEN it SHALL follow the same signature pattern as `compress_zstd_into()` and `compress_lz4_into()`.
3. WHEN `CompressionScratch` is used THEN it SHALL support bzip2 compression via the new function.
4. WHEN the function is implemented THEN it SHALL have corresponding unit tests.

---

### Requirement 3: Add `compress_xz_into()` Function

**User Story:** As a developer using buffer reuse patterns, I want a `compress_xz_into()` function, so that I can compress XZ data without allocating new buffers each time.

#### Acceptance Criteria

1. WHEN `compress_xz_into(data, level, output)` is called THEN it SHALL compress data into the provided output buffer.
2. WHEN the function is implemented THEN it SHALL follow the same signature pattern as other `_into()` variants.
3. WHEN `CompressionScratch` is used THEN it SHALL support XZ compression via the new function.
4. WHEN the function is implemented THEN it SHALL have corresponding unit tests.

**Note:** The lzma-rs library may have limitations; document if full feature parity is not possible.

---

### Requirement 4: Fix Integer Overflow Protection in Chunking

**User Story:** As a developer, I want chunking size parameters to be safely converted to u32, so that overflow bugs cannot cause undefined behavior.

#### Acceptance Criteria

1. WHEN `ChunkingOptions` values are passed to FastCDC THEN the conversion from usize to u32 SHALL use explicit checked conversion with proper error handling.
2. WHEN a value exceeds `u32::MAX` THEN it SHALL return `ChunkingError::InvalidOptions` with a descriptive message.
3. WHEN the fix is applied THEN existing validation (max <= 1GB) SHALL remain as defense-in-depth.

**Location:** `src/chunking.rs:214-216`

---

### Requirement 5: Improve Atomic Counter Ordering

**User Story:** As a developer, I want trace sampling to use correct memory ordering, so that concurrent access produces predictable sampling behavior.

#### Acceptance Criteria

1. WHEN `TRACE_SAMPLE_COUNTER` is incremented THEN it SHALL use `Ordering::AcqRel` or document why `Relaxed` is acceptable for approximate sampling.
2. WHEN the decision is to keep `Relaxed` THEN a code comment SHALL explain the trade-off.

**Location:** `src/chunking.rs:502-503, 719`

---

### Requirement 6: Add Blake3 Hash Direct Test

**User Story:** As a maintainer, I want direct unit tests for `blake3_hash()`, so that the function is explicitly covered by the test suite.

#### Acceptance Criteria

1. WHEN `blake3_hash()` is called THEN a unit test SHALL verify correct output for known input.
2. WHEN the test is added THEN it SHALL be in the hashing module's test section.

**Location:** `src/hashing.rs`

---

### Requirement 7: Add Edge Case Tests

**User Story:** As a maintainer, I want comprehensive edge case tests, so that boundary conditions are verified.

#### Acceptance Criteria

1. WHEN zero-length data is chunked THEN the behavior SHALL be tested and documented.
2. WHEN data at exact chunk size boundaries is processed THEN behavior SHALL be tested.
3. WHEN compression/decompression handles empty input THEN behavior SHALL be tested.

---

### Requirement 8: Document BlockingAsyncReadAdapter Risks

**User Story:** As a developer using async features, I want clear documentation about `BlockingAsyncReadAdapter` risks, so that I avoid deadlocks.

#### Acceptance Criteria

1. WHEN `BlockingAsyncReadAdapter` is used THEN documentation SHALL clearly warn about blocking behavior.
2. WHEN misuse is detected at runtime (if possible) THEN a debug assertion or log warning SHALL alert the developer.

**Location:** `src/chunking.rs:807-835`

---

## Non-Functional Requirements

### NFR-1: Backward Compatibility

1. All changes SHALL maintain backward compatibility with existing public API.
2. No breaking changes to function signatures or behavior (except bug fixes).

### NFR-2: Test Coverage

1. All new code SHALL have corresponding unit tests.
2. All bug fixes SHALL have regression tests.

### NFR-3: Performance

1. Fixes SHALL NOT degrade performance by more than 1%.
2. New `_into()` functions SHALL provide measurable allocation reduction.

---

## Out of Scope

1. Cargo.toml edition change (already correct - "2024" is valid for nightly)
2. SIMD optimizations for base32 (low priority, profile first)
3. Metrics batching (low priority)
4. Async runtime mixing guards (separate concern)
