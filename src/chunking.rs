#![allow(clippy::cast_precision_loss)]
#[cfg(feature = "async-stream")]
use async_stream::try_stream;
use blake3;
use bytes::{Bytes, BytesMut};
use fastcdc::v2020::FastCDC;
#[cfg(feature = "async-stream")]
use futures::executor::block_on;
#[cfg(feature = "async-stream")]
use futures::io::AsyncRead;
#[cfg(feature = "async-stream")]
use futures::io::AsyncReadExt;
#[cfg(feature = "async-stream")]
use futures::stream::Stream;
#[cfg(feature = "async-stream")]
use futures::stream::StreamExt;
use metrics::{counter, histogram};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::env;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::io::{ErrorKind, Read};
#[cfg(feature = "async-stream")]
use std::pin::Pin;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "async-stream")]
use std::task::{Context, Poll};
use tracing::{debug, instrument, trace};

#[derive(Debug, thiserror::Error)]
pub enum ChunkingError {
    #[error(
        "bounds_check_failed: offset {offset} + length {length} exceeds data length {data_len}"
    )]
    Bounds {
        data_len: usize,
        offset: usize,
        length: usize,
    },
    #[error("io_error: {0}")]
    Io(#[from] std::io::Error),
    #[error("zero_length_chunk: FastCDC returned a chunk with length 0")]
    ZeroLengthChunk,
    #[error("invalid_chunking_options: {0}")]
    InvalidOptions(String),
    #[error("buffer_limit_exceeded: attempted {attempted} bytes, limit {limit} bytes")]
    BufferLimitExceeded { attempted: usize, limit: usize },
}

/// Hash algorithm used when producing chunk metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HashAlgorithm {
    Blake3,
    Sha256,
}

/// Compute hash of data using the specified algorithm.
///
/// This function is marked `#[inline(always)]` to eliminate match overhead
/// in the hot path. The compiler will inline the function at each call site,
/// allowing branch prediction to work optimally when the algorithm is known.
#[inline]
fn compute_hash(data: &[u8], algorithm: HashAlgorithm) -> [u8; 32] {
    match algorithm {
        HashAlgorithm::Blake3 => {
            let hash = blake3::hash(data);
            *hash.as_bytes()
        }
        HashAlgorithm::Sha256 => crate::hashing::sha256_hash_raw(data),
    }
}

/// Metadata for a single chunk emitted by streaming chunkers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Hash of the chunk payload (32 bytes) using the configured [`HashAlgorithm`].
    /// Using raw bytes is more efficient than hex strings for storage/transmission.
    #[serde(with = "hex_serde")]
    pub hash: [u8; 32],
    /// Starting byte offset of the chunk relative to the reader.
    pub offset: u64,
    /// Chunk length in bytes.
    pub length: usize,
    /// The actual chunk data.
    /// Using `Bytes` allows zero-copy cloning and efficient integration with S3 SDKs.
    #[serde(skip)]
    pub payload: Bytes,
}

impl ChunkMetadata {
    /// Returns the hash as a hex string.
    pub fn hash_hex(&self) -> String {
        hex::encode(self.hash)
    }
}

/// Metadata for a chunk when callers do not need the payload bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkDescriptor {
    /// Hash of the chunk payload (32 bytes) using the configured [`HashAlgorithm`].
    #[serde(with = "hex_serde")]
    pub hash: [u8; 32],
    /// Starting byte offset of the chunk relative to the input.
    pub offset: u64,
    /// Chunk length in bytes.
    pub length: usize,
}

impl ChunkDescriptor {
    /// Returns the hash as a hex string.
    #[must_use]
    pub fn hash_hex(&self) -> String {
        hex::encode(self.hash)
    }
}

impl PartialEq for ChunkMetadata {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && self.offset == other.offset && self.length == other.length
    }
}

impl Eq for ChunkMetadata {}

impl Hash for ChunkMetadata {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
        self.offset.hash(state);
        self.length.hash(state);
    }
}

mod hex_serde {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8; 32], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&hex::encode(bytes))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 32], D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let mut bytes = [0u8; 32];
        hex::decode_to_slice(s, &mut bytes).map_err(serde::de::Error::custom)?;
        Ok(bytes)
    }
}

/// Configurable bounds for `FastCDC` chunking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkingOptions {
    /// Minimum chunk size in bytes.
    pub min_size: usize,
    /// Average (target) chunk size in bytes.
    pub avg_size: usize,
    /// Maximum chunk size in bytes.
    pub max_size: usize,
}

impl ChunkingOptions {
    /// # Errors
    ///
    /// Returns `ChunkingError::InvalidOptions` if the provided sizes are invalid (e.g. min > max).
    #[instrument(level = "debug", skip(min_size, avg_size, max_size), fields(min = ?min_size, avg = ?avg_size, max = ?max_size))]
    pub fn resolve(
        min_size: Option<usize>,
        avg_size: Option<usize>,
        max_size: Option<usize>,
    ) -> Result<Self, ChunkingError> {
        let options = Self {
            min_size: min_size.unwrap_or(256 * 1024),      // 256 KB
            avg_size: avg_size.unwrap_or(1024 * 1024),     // 1 MB
            max_size: max_size.unwrap_or(4 * 1024 * 1024), // 4 MB
        };

        options.validate()?;

        trace!(?options, "resolved_chunking_options");
        Ok(options)
    }

    fn validate(&self) -> Result<(), ChunkingError> {
        if self.min_size < 64 {
            return Err(ChunkingError::InvalidOptions(
                "min_size must be >= 64".into(),
            ));
        }
        if self.min_size > self.avg_size {
            return Err(ChunkingError::InvalidOptions(
                "min_size must be <= avg_size".into(),
            ));
        }
        if self.avg_size > self.max_size {
            return Err(ChunkingError::InvalidOptions(
                "avg_size must be <= max_size".into(),
            ));
        }
        if self.max_size > 1024 * 1024 * 1024 {
            return Err(ChunkingError::InvalidOptions(
                "max_size must be <= 1GB".into(),
            ));
        }

        Ok(())
    }
}

/// Chunk data using `FastCDC` (Content-Defined Chunking)
/// Args: data (binary), `min_size` (optional), `avg_size` (optional), `max_size` (optional)
/// Returns: list of `ChunkMetadata`
///
/// # Errors
///
/// Returns `ChunkingError` if reading from the data fails (unlikely for in-memory data).
#[instrument(skip(data), fields(data_len = data.len()))]
pub fn chunk_data(
    data: &[u8],
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
) -> Result<Vec<ChunkMetadata>, ChunkingError> {
    chunk_data_with_hash(data, min_size, avg_size, max_size, HashAlgorithm::Sha256)
}

/// Same as [`chunk_data`] but lets callers choose the hash algorithm.
///
/// # Errors
///
/// Returns `ChunkingError` if reading from the data fails.
pub fn chunk_data_with_hash(
    data: &[u8],
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
    hash: HashAlgorithm,
) -> Result<Vec<ChunkMetadata>, ChunkingError> {
    let descriptors = chunk_descriptors_with_hash(data, min_size, avg_size, max_size, hash)?;

    descriptors
        .into_iter()
        .map(|descriptor| {
            let offset = usize::try_from(descriptor.offset).map_err(|_| ChunkingError::Bounds {
                data_len: data.len(),
                offset: usize::MAX,
                length: descriptor.length,
            })?;
            let end = offset
                .checked_add(descriptor.length)
                .ok_or(ChunkingError::Bounds {
                    data_len: data.len(),
                    offset,
                    length: descriptor.length,
                })?;

            if end > data.len() {
                return Err(ChunkingError::Bounds {
                    data_len: data.len(),
                    offset,
                    length: descriptor.length,
                });
            }

            Ok(ChunkMetadata {
                hash: descriptor.hash,
                offset: descriptor.offset,
                length: descriptor.length,
                payload: Bytes::copy_from_slice(&data[offset..end]),
            })
        })
        .collect()
}

/// Chunk data and return only hash/offset/length metadata.
///
/// This is the preferred API for NIF callers that already own the source binary
/// and can slice it in the host runtime.
///
/// # Errors
///
/// Returns `ChunkingError` when chunking options are invalid or `FastCDC` returns
/// invalid bounds.
pub fn chunk_descriptors(
    data: &[u8],
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
) -> Result<Vec<ChunkDescriptor>, ChunkingError> {
    chunk_descriptors_with_hash(data, min_size, avg_size, max_size, HashAlgorithm::Sha256)
}

/// Same as [`chunk_descriptors`] but lets callers choose the hash algorithm.
///
/// # Errors
///
/// Returns `ChunkingError` when chunking options are invalid or `FastCDC` returns
/// invalid bounds.
pub fn chunk_descriptors_with_hash(
    data: &[u8],
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
    hash: HashAlgorithm,
) -> Result<Vec<ChunkDescriptor>, ChunkingError> {
    let options = ChunkingOptions::resolve(min_size, avg_size, max_size)?;

    // 1. FastCDC pass. Observed throughput is host and corpus dependent.
    // We collect cut points first to enable parallel hashing.
    let chunker = FastCDC::new(data, options.min_size, options.avg_size, options.max_size);

    let cut_points: Vec<_> = chunker.collect();

    // 2. Hashing Pass (Parallel, CPU-Bound)
    // Rayon will distribute the hashing of chunks across all available cores.
    let chunks: Result<Vec<ChunkDescriptor>, ChunkingError> = cut_points
        .par_iter()
        .map(|chunk_def| {
            let offset = chunk_def.offset;
            let length = chunk_def.length;

            // Safety check to prevent panics in the thread pool
            if offset
                .checked_add(length)
                .is_none_or(|end| end > data.len())
            {
                return Err(ChunkingError::Bounds {
                    data_len: data.len(),
                    offset,
                    length,
                });
            }

            let chunk_slice = &data[offset..offset + length];

            let hash_array = compute_hash(chunk_slice, hash);

            Ok(ChunkDescriptor {
                hash: hash_array,
                offset: offset as u64,
                length,
            })
        })
        .collect();

    let chunks = chunks?;

    debug!(chunk_count = chunks.len(), "chunking_complete");
    Ok(chunks)
}

/// A streaming chunk reader that yields chunk metadata and hashes without buffering
/// the entire source in memory.
///
/// This implementation uses a "slab" buffer strategy (`BytesMut`) to minimize memory allocations.
/// Instead of allocating a new `Vec<u8>` for every chunk, it reads large blocks into a shared
/// buffer and yields zero-copy `Bytes` handles (slices) to the consumer.
pub struct ChunkStream<R: Read> {
    reader: R,
    buffer: BytesMut,
    min_size: usize,
    #[allow(dead_code)]
    avg_size: usize,
    max_size: usize,
    hash: HashAlgorithm,
    position: u64,
    eof: bool,
    pending_chunks: VecDeque<Result<ChunkMetadata, ChunkingError>>,
}

static TRACE_SAMPLE_COUNTER: AtomicU64 = AtomicU64::new(0);
const TRACE_SAMPLE_EVERY: u64 = 1024;

/// Cached metric handles to eliminate per-chunk metric name lookups.
/// Using `OnceLock` ensures thread-safe one-time initialization.
struct CachedMetrics {
    chunks_emitted: metrics::Counter,
    bytes_processed: metrics::Counter,
    chunk_size: metrics::Histogram,
}

static METRICS: OnceLock<CachedMetrics> = OnceLock::new();

/// Returns cached metric handles, initializing them on first call.
/// This eliminates the overhead of metric name lookups on each chunk emission.
fn get_metrics() -> &'static CachedMetrics {
    METRICS.get_or_init(|| CachedMetrics {
        chunks_emitted: counter!("chunker.chunks_emitted"),
        bytes_processed: counter!("chunker.bytes_processed"),
        chunk_size: histogram!("chunker.chunk_size"),
    })
}
const DEFAULT_READ_SLICE_CAP: usize = 8 * 1024 * 1024; // 8 MiB per read keeps zero-fill overhead bounded
const MAX_READ_SLICE_CAP: usize = 256 * 1024 * 1024; // 256 MiB hard ceiling
const MIN_READ_SLICE_CAP: usize = 4096;
#[cfg(feature = "async-stream")]
const DEFAULT_ASYNC_BUFFER_LIMIT: usize = 2 * 1024 * 1024 * 1024; // 2 GiB safety cap, overridable for binary caches
#[cfg(feature = "async-stream")]
const MAX_ASYNC_BUFFER_LIMIT: usize = 3 * 1024 * 1024 * 1024; // upper clamp to avoid runaway allocations
#[cfg(feature = "async-stream")]
const MIN_ASYNC_BUFFER_LIMIT: usize = 64 * 1024 * 1024; // 64 MiB minimum to keep streaming efficient

fn effective_read_slice_cap() -> usize {
    let from_env = env::var("CHUNKER_READ_SLICE_CAP_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| (MIN_READ_SLICE_CAP..=MAX_READ_SLICE_CAP).contains(&v));
    from_env.unwrap_or(DEFAULT_READ_SLICE_CAP)
}

#[cfg(feature = "async-stream")]
fn effective_async_buffer_limit() -> usize {
    let from_env = env::var("CHUNKER_ASYNC_BUFFER_LIMIT_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    let cap = from_env.unwrap_or(DEFAULT_ASYNC_BUFFER_LIMIT);
    cap.clamp(
        MIN_ASYNC_BUFFER_LIMIT,
        MAX_ASYNC_BUFFER_LIMIT.min(usize::MAX),
    )
}

impl<R: Read> fmt::Debug for ChunkStream<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChunkStream")
            .field("position", &self.position)
            .field("buffer_len", &self.buffer.len())
            .field("eof", &self.eof)
            .finish_non_exhaustive()
    }
}

impl<R: Read> ChunkStream<R> {
    /// Create a streaming chunker around any [`std::io::Read`] implementation.
    /// Uses `FastCDC` with optional min/avg/max overrides.
    ///
    /// # Errors
    ///
    /// Returns `ChunkingError::InvalidOptions` if the provided sizes are invalid.
    #[instrument(skip(reader))]
    pub fn new(
        reader: R,
        min_size: Option<usize>,
        avg_size: Option<usize>,
        max_size: Option<usize>,
    ) -> Result<Self, ChunkingError> {
        Self::new_with_hash(reader, min_size, avg_size, max_size, HashAlgorithm::Sha256)
    }

    /// Create a streaming chunker with an explicit hash algorithm.
    ///
    /// This initializes the internal slab buffer lazily to avoid large allocations upfront.
    ///
    /// # Errors
    ///
    /// Returns `ChunkingError::InvalidOptions` if the provided sizes are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use chunker::chunking::{ChunkStream, HashAlgorithm};
    /// use std::io::Cursor;
    ///
    /// let data = b"some data to chunk";
    /// let stream = ChunkStream::new_with_hash(
    ///     Cursor::new(data),
    ///     Some(1024),
    ///     Some(4096),
    ///     Some(8192),
    ///     HashAlgorithm::Blake3
    /// )?;
    ///
    /// for chunk in stream {
    ///     let chunk = chunk?;
    ///     println!("Chunk: {} bytes", chunk.length);
    /// }
    /// # Ok::<(), chunker::chunking::ChunkingError>(())
    /// ```
    #[instrument(skip(reader))]
    pub fn new_with_hash(
        reader: R,
        min_size: Option<usize>,
        avg_size: Option<usize>,
        max_size: Option<usize>,
        hash: HashAlgorithm,
    ) -> Result<Self, ChunkingError> {
        let options = ChunkingOptions::resolve(min_size, avg_size, max_size)?;

        // Start with a small buffer (e.g. min_size) and let BytesMut grow as needed.
        // We cap the initial allocation to avoid OOM on creation if max_size is huge.
        let initial_capacity = std::cmp::min(options.min_size, 64 * 1024);
        let buffer = BytesMut::with_capacity(initial_capacity);

        Ok(Self {
            reader,
            buffer,
            min_size: options.min_size,
            avg_size: options.avg_size,
            max_size: options.max_size,
            hash,
            position: 0,
            eof: false,
            pending_chunks: VecDeque::new(),
        })
    }
}

impl<R: Read> Iterator for ChunkStream<R> {
    type Item = Result<ChunkMetadata, ChunkingError>;

    #[allow(clippy::too_many_lines)]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(chunk) = self.pending_chunks.pop_front() {
            return Some(chunk);
        }

        loop {
            // 1. Try to find chunks in the current buffer
            // Early-exit: skip FastCDC iteration if buffer is smaller than min_size and not at EOF
            // FastCDC cannot produce a chunk smaller than min_size unless it's the final chunk.
            if !self.buffer.is_empty() && (self.buffer.len() >= self.min_size || self.eof) {
                // Create a FastCDC iterator over the buffer using validated params.
                let cdc = FastCDC::new(&self.buffer, self.min_size, self.avg_size, self.max_size);

                // Collect all valid cut points in the current buffer
                let mut cut_points = Vec::new();
                let mut total_len = 0;

                for chunk in cdc {
                    let len = chunk.length;
                    let offset = chunk.offset;

                    // Safety check: FastCDC should never return a zero-length chunk
                    if len == 0 {
                        return Some(Err(ChunkingError::ZeroLengthChunk));
                    }

                    // Safety check: FastCDC should start at 0 for the first chunk in the slice
                    // Subsequent chunks will have offset > 0 relative to the start of the buffer
                    if cut_points.is_empty() && offset != 0 {
                        return Some(Err(std::io::Error::new(
                            ErrorKind::InvalidData,
                            format!("FastCDC returned non-zero offset {offset} for first chunk"),
                        )
                        .into()));
                    }

                    // Check if this chunk is "complete"
                    // If it touches the end of the buffer, and we are not at EOF, and it's smaller than max_size,
                    // it might be a partial chunk (FastCDC didn't find a cut point yet).
                    let touches_end = offset + len == self.buffer.len();
                    if touches_end && !self.eof && len < self.max_size {
                        break;
                    }

                    cut_points.push(chunk);
                    total_len += len;
                }

                if !cut_points.is_empty() {
                    // Extract the data for all valid chunks at once
                    // This advances the buffer by total_len
                    let batch_data = self.buffer.split_to(total_len).freeze();

                    // Process chunks in parallel (hashing)
                    // We map the cut points to ChunkMetadata
                    let current_position = self.position;
                    let hash_algo = self.hash;

                    // Helper closure for processing a single chunk
                    let process_chunk = |chunk: &fastcdc::v2020::Chunk| {
                        let len = chunk.length;
                        let offset = chunk.offset;

                        // Safety check with overflow protection
                        if offset
                            .checked_add(len)
                            .is_none_or(|end| end > batch_data.len())
                        {
                            return Err(ChunkingError::Bounds {
                                data_len: batch_data.len(),
                                offset,
                                length: len,
                            });
                        }

                        let chunk_slice = batch_data.slice(offset..offset + len);

                        // Metrics (thread-safe, using cached handles)
                        let m = get_metrics();
                        m.chunks_emitted.increment(1);
                        m.bytes_processed.increment(len as u64);
                        m.chunk_size.record(len as f64);

                        let hash_array = compute_hash(&chunk_slice, hash_algo);

                        let chunk_offset = current_position + offset as u64;

                        if tracing::enabled!(tracing::Level::TRACE)
                            && TRACE_SAMPLE_COUNTER
                                .fetch_add(1, Ordering::Relaxed)
                                .is_multiple_of(TRACE_SAMPLE_EVERY)
                        {
                            trace!(offset = chunk_offset, length = len, "chunk_emitted");
                        }

                        Ok(ChunkMetadata {
                            hash: hash_array,
                            offset: chunk_offset,
                            length: len,
                            payload: chunk_slice,
                        })
                    };

                    let chunks: Vec<Result<ChunkMetadata, ChunkingError>> = if cut_points.len() > 4
                    {
                        cut_points.par_iter().map(process_chunk).collect()
                    } else {
                        cut_points.iter().map(process_chunk).collect()
                    };

                    // Update position
                    self.position += total_len as u64;

                    // Push to pending queue
                    self.pending_chunks.extend(chunks);

                    // Return the first one immediately
                    return self.pending_chunks.pop_front();
                }
            }

            // 2. If no chunk found, or buffer empty, we need more data.
            if self.eof {
                // If we are at EOF and still have data in buffer, it means FastCDC didn't find a cut point.
                // This is the last chunk (remainder).
                if self.buffer.is_empty() {
                    return None;
                }

                let len = self.buffer.len();
                let chunk_data = self.buffer.split_to(len).freeze();

                // Metrics (using cached handles)
                let m = get_metrics();
                m.chunks_emitted.increment(1);
                m.bytes_processed.increment(len as u64);
                m.chunk_size.record(len as f64);

                let hash_array = compute_hash(&chunk_data, self.hash);

                let chunk_offset = self.position;
                self.position += len as u64;

                return Some(Ok(ChunkMetadata {
                    hash: hash_array,
                    offset: chunk_offset,
                    length: len,
                    payload: chunk_data,
                }));
            }

            // 3. Read more data (cap per-read to avoid huge allocations)
            // We increase the read size to encourage batching (e.g. 8x max_size)
            let slice_cap = effective_read_slice_cap();
            let target_read = std::cmp::max(self.max_size * 8, 4 * 1024 * 1024);
            let read_size =
                std::cmp::max(std::cmp::min(target_read, slice_cap), MIN_READ_SLICE_CAP);

            // Reserve space
            self.buffer.reserve(read_size);

            // Read into the buffer
            // Using resize to zero-init (safe)
            let start = self.buffer.len();
            self.buffer.resize(start + read_size, 0);

            match self.reader.read(&mut self.buffer[start..]) {
                Ok(0) => {
                    self.eof = true;
                    self.buffer.truncate(start); // Remove the extra zeros
                    // Loop again to process remainder
                }
                Ok(n) => {
                    self.buffer.truncate(start + n); // Keep only what we read
                    // Loop again to process
                }
                Err(e) => {
                    if e.kind() == ErrorKind::Interrupted {
                        self.buffer.truncate(start);
                        continue;
                    }
                    return Some(Err(ChunkingError::Io(e)));
                }
            }
        }
    }
}

/// A streaming chunk reader that yields chunk metadata and hashes without buffering
/// the entire source in memory.
///
/// This implementation uses a "slab" buffer strategy (`BytesMut`) to minimize memory allocations.
#[cfg(feature = "async-stream")]
pub struct ChunkStreamAsync<R> {
    stream: Pin<Box<dyn Stream<Item = Result<ChunkMetadata, ChunkingError>> + Send>>,
    _phantom: std::marker::PhantomData<R>,
}

#[cfg(feature = "async-stream")]
impl<R> fmt::Debug for ChunkStreamAsync<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChunkStreamAsync").finish_non_exhaustive()
    }
}

#[cfg(feature = "async-stream")]
impl<R> Unpin for ChunkStreamAsync<R> {}

#[cfg(feature = "async-stream")]
impl<R: AsyncRead + Unpin + Send + 'static> ChunkStreamAsync<R> {
    /// Create a streaming chunker around any [`futures::io::AsyncRead`] implementation.
    ///
    /// # Errors
    ///
    /// Returns `ChunkingError::InvalidOptions` if the provided sizes are invalid.
    pub fn new(
        reader: R,
        min_size: Option<usize>,
        avg_size: Option<usize>,
        max_size: Option<usize>,
    ) -> Result<Self, ChunkingError> {
        Self::new_with_hash(reader, min_size, avg_size, max_size, HashAlgorithm::Sha256)
    }

    /// Create a streaming chunker with an explicit hash algorithm.
    ///
    /// # Errors
    ///
    /// Returns `ChunkingError::InvalidOptions` if the provided sizes are invalid.
    #[allow(clippy::too_many_lines)]
    pub fn new_with_hash(
        mut reader: R,
        min_size: Option<usize>,
        avg_size: Option<usize>,
        max_size: Option<usize>,
        hash: HashAlgorithm,
    ) -> Result<Self, ChunkingError> {
        let options = ChunkingOptions::resolve(min_size, avg_size, max_size)?;

        // Start with a small buffer (e.g. min_size) and let BytesMut grow as needed.
        // We cap the initial allocation to avoid OOM on creation if max_size is huge.
        let initial_capacity = std::cmp::min(options.min_size, 64 * 1024);
        let mut buffer = BytesMut::with_capacity(initial_capacity);
        let mut position = 0u64;
        let mut eof = false;

        let stream = try_stream! {
            loop {
                // 1. Try to find a chunk in the current buffer
                if !buffer.is_empty() {
                    let mut cdc = FastCDC::new(
                        &buffer,
                        options.min_size,
                        options.avg_size,
                        options.max_size,
                    );

                    if let Some(chunk) = cdc.next() {
                        let len = chunk.length;
                        let offset = chunk.offset;

                        if len == 0 {
                            Err(ChunkingError::ZeroLengthChunk)?;
                        }

                        if offset != 0 {
                            Err(std::io::Error::new(
                                ErrorKind::InvalidData,
                                format!("FastCDC returned non-zero offset {offset} for first chunk"),
                            ))?;
                        }

                        if len > buffer.len() {
                            Err(ChunkingError::Bounds {
                                data_len: buffer.len(),
                                offset: 0,
                                length: len,
                            })?;
                        }

                        if len != buffer.len() || eof || len >= options.max_size {
                            let chunk_data = buffer.split_to(len).freeze();

                            let m = get_metrics();
                            m.chunks_emitted.increment(1);
                            m.bytes_processed.increment(len as u64);
                            m.chunk_size.record(len as f64);

                            let hash_array = compute_hash(&chunk_data, hash);

                            let chunk_offset = position;
                            position += len as u64;

                            if tracing::enabled!(tracing::Level::TRACE)
                                && TRACE_SAMPLE_COUNTER.fetch_add(1, Ordering::Relaxed).is_multiple_of(TRACE_SAMPLE_EVERY)
                            {
                                trace!(offset = chunk_offset, length = len, "chunk_emitted");
                            }

                            yield ChunkMetadata {
                                hash: hash_array,
                                offset: chunk_offset,
                                length: len,
                                payload: chunk_data,
                            };
                            continue;
                        }
                    }
                }

                // 2. If no chunk found, or buffer empty, we need more data.
                if eof {
                    if buffer.is_empty() {
                        break;
                    }

                    let len = buffer.len();
                    let chunk_data = buffer.split_to(len).freeze();

                    let m = get_metrics();
                    m.chunks_emitted.increment(1);
                    m.bytes_processed.increment(len as u64);
                    m.chunk_size.record(len as f64);

                    let hash_array = compute_hash(&chunk_data, hash);

                    let chunk_offset = position;
                    position += len as u64;

                    yield ChunkMetadata {
                        hash: hash_array,
                        offset: chunk_offset,
                        length: len,
                        payload: chunk_data,
                    };
                    continue;
                }

                // 3. Read more data (cap per-read to avoid huge allocations)
                let slice_cap = effective_read_slice_cap();
                let read_size = std::cmp::max(std::cmp::min(options.max_size, slice_cap), MIN_READ_SLICE_CAP);
                buffer.reserve(read_size);
                let start = buffer.len();
                buffer.resize(start + read_size, 0);

                match reader.read(&mut buffer[start..]).await {
                    Ok(0) => {
                        eof = true;
                        buffer.truncate(start);
                    }
                    Ok(n) => {
                        buffer.truncate(start + n);
                    }
                    Err(e) => {
                        Err(ChunkingError::Io(e))?;
                    }
                }
            }
        };

        Ok(Self {
            stream: Box::pin(stream),
            _phantom: std::marker::PhantomData,
        })
    }
}

#[cfg(feature = "async-stream")]
impl<R> Stream for ChunkStreamAsync<R> {
    type Item = Result<ChunkMetadata, ChunkingError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.as_mut().get_mut().stream.as_mut().poll_next(cx)
    }
}

/// Adapter to allow [`ChunkStream`] usage with asynchronous readers.
///
/// # Warning
///
/// This adapter uses `block_on` to bridge async reads to synchronous reads.
/// This will **block the current thread** during reads.
///
/// - If used inside an async runtime (like Tokio), this may block the reactor or panic.
/// - Only use this with `futures` executors or when you are sure blocking is safe.
#[cfg(feature = "async-stream")]
pub struct BlockingAsyncReadAdapter<R: AsyncRead + Unpin> {
    inner: R,
}

#[cfg(feature = "async-stream")]
impl<R: AsyncRead + Unpin> fmt::Debug for BlockingAsyncReadAdapter<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BlockingAsyncReadAdapter").finish()
    }
}

#[cfg(feature = "async-stream")]
impl<R: AsyncRead + Unpin> Read for BlockingAsyncReadAdapter<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        // Small, blocking bridge for FastCDC; consider calling from a blocking task.
        // WARNING: This blocks the thread!
        block_on(self.inner.read(buf))
    }
}

/// Construct chunks from an asynchronous reader by streaming the input.
///
/// # Warning: Memory Usage
///
/// This function collects all chunks into a `Vec`, which means the entire payload
/// will eventually reside in memory. To avoid OOM on large files, use [`ChunkStreamAsync`] directly.
///
/// This function enforces a buffer limit on the total accumulated payload size.
///
/// # Errors
///
/// Returns `ChunkingError` if reading from the stream fails or if chunking parameters are invalid.
#[cfg(feature = "async-stream")]
pub async fn chunk_data_async<R: AsyncRead + Unpin + Send + 'static>(
    reader: R,
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
) -> Result<Vec<ChunkMetadata>, ChunkingError> {
    let buffer_limit = effective_async_buffer_limit();
    let mut stream = ChunkStreamAsync::new(reader, min_size, avg_size, max_size)?;
    let mut chunks = Vec::new();
    let mut total_len = 0;

    while let Some(chunk_res) = stream.next().await {
        let chunk = chunk_res?;
        total_len += chunk.length;

        if total_len > buffer_limit {
            return Err(ChunkingError::BufferLimitExceeded {
                attempted: total_len,
                limit: buffer_limit,
            });
        }

        chunks.push(chunk);
    }

    Ok(chunks)
}

/// Stream chunks from an asynchronous reader without buffering the entire input.
///
/// # Warning: Blocking Operation
///
/// This function bridges async reads into FastCDC’s synchronous streamer.
/// It performs **blocking reads** internally using `block_on`.
///
/// - **Do not call this directly in an async task** (e.g., `tokio::spawn`). It will block the thread.
/// - If using Tokio, wrap this in `tokio::task::spawn_blocking`.
/// - This may panic if the underlying `AsyncRead` requires a reactor that is currently blocked.
///
/// # Errors
///
/// Returns `ChunkingError` if reading from the stream fails or if chunking parameters are invalid.
#[cfg(feature = "async-stream")]
pub fn chunk_stream_blocking_adapter<R: AsyncRead + Unpin>(
    reader: R,
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
) -> Result<Vec<ChunkMetadata>, ChunkingError> {
    let adapter = BlockingAsyncReadAdapter { inner: reader };
    chunk_stream(adapter, min_size, avg_size, max_size)
}

/// Async-friendly wrapper that uses the non-blocking `ChunkStreamAsync`.
///
/// This function is equivalent to `chunk_data_async` but kept for backward compatibility.
///
/// # Errors
///
/// Returns `ChunkingError` if reading from the stream fails or if chunking parameters are invalid.
#[cfg(feature = "async-stream")]
pub async fn chunk_stream_async<R: AsyncRead + Unpin + Send + 'static>(
    reader: R,
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
) -> Result<Vec<ChunkMetadata>, ChunkingError> {
    chunk_data_async(reader, min_size, avg_size, max_size).await
}

/// Convenience function to chunk a stream and collect all chunk metadata into a vector.
///
/// This is useful when you want to process a stream but need all chunk metadata at once.
///
/// # Errors
///
/// Returns `ChunkingError` if reading from the stream fails.
pub fn chunk_stream<R: Read>(
    reader: R,
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
) -> Result<Vec<ChunkMetadata>, ChunkingError> {
    chunk_stream_with_hash(reader, min_size, avg_size, max_size, HashAlgorithm::Sha256)
}

/// Same as [`chunk_stream`] but lets callers choose the hash algorithm.
///
/// # Errors
///
/// Returns `ChunkingError` if reading from the stream fails.
pub fn chunk_stream_with_hash<R: Read>(
    reader: R,
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
    hash: HashAlgorithm,
) -> Result<Vec<ChunkMetadata>, ChunkingError> {
    let stream = ChunkStream::new_with_hash(reader, min_size, avg_size, max_size, hash)?;
    let mut chunks = Vec::with_capacity(128);
    for chunk in stream {
        chunks.push(chunk?);
    }
    Ok(chunks)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};

    #[test]
    fn streaming_reader_emits_full_length() -> Result<(), ChunkingError> {
        let size = 3 * 1024 * 1024 + 321; // Just over 3 MiB
        let data = vec![42_u8; size];
        let cursor = Cursor::new(&data);
        let reader = BufReader::new(cursor);

        let stream = ChunkStream::new(reader, Some(16 * 1024), Some(32 * 1024), Some(64 * 1024))?;
        let mut total = 0_usize;
        let mut count = 0_usize;

        for chunk in stream {
            let chunk = chunk?;
            total += chunk.length;
            count += 1;
        }

        assert_eq!(total, size);
        assert!(count > 1); // Ensure multiple chunks were produced for large data

        Ok(())
    }

    #[test]
    fn chunking_options_defaults_match_chunk_data() -> Result<(), ChunkingError> {
        let payload = b"streaming chunker parity test".repeat(2048);
        let mut streaming = ChunkStream::new(Cursor::new(&payload), None, None, None)?;
        let collected: Vec<_> = streaming.by_ref().collect::<Result<_, _>>()?;

        let eager = chunk_data(&payload, None, None, None)?;
        assert_eq!(collected, eager);

        Ok(())
    }

    #[test]
    fn test_chunk_data_basic() -> Result<(), ChunkingError> {
        let data = vec![0u8; 1024 * 1024]; // 1MB
        let chunks = chunk_data(&data, None, None, None)?;

        assert!(!chunks.is_empty());

        let mut total_len = 0;
        for chunk in &chunks {
            assert_eq!(chunk.offset as usize, total_len);
            total_len += chunk.length;
        }
        assert_eq!(total_len, data.len());
        Ok(())
    }

    #[test]
    fn test_chunk_data_small() -> Result<(), ChunkingError> {
        let data = b"small data";
        let chunks = chunk_data(data, None, None, None)?;

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].offset, 0);
        assert_eq!(chunks[0].length, data.len());
        Ok(())
    }

    #[test]
    fn test_chunk_data_with_blake3() -> Result<(), ChunkingError> {
        let data = vec![7u8; 4096];
        let sha_chunks = chunk_data_with_hash(&data, None, None, None, HashAlgorithm::Sha256)?;
        let blake_chunks = chunk_data_with_hash(&data, None, None, None, HashAlgorithm::Blake3)?;

        assert_eq!(sha_chunks.len(), blake_chunks.len());
        // Hashes should differ across algorithms
        assert_ne!(sha_chunks[0].hash, blake_chunks[0].hash);
        Ok(())
    }

    #[test]
    fn test_chunk_descriptors_match_chunk_metadata() -> Result<(), ChunkingError> {
        let data = b"descriptor parity test payload".repeat(8192);
        let descriptors = chunk_descriptors(&data, Some(1024), Some(4096), Some(8192))?;
        let chunks = chunk_data(&data, Some(1024), Some(4096), Some(8192))?;

        assert_eq!(descriptors.len(), chunks.len());

        for (descriptor, chunk) in descriptors.iter().zip(chunks.iter()) {
            assert_eq!(descriptor.hash, chunk.hash);
            assert_eq!(descriptor.hash_hex(), chunk.hash_hex());
            assert_eq!(descriptor.offset, chunk.offset);
            assert_eq!(descriptor.length, chunk.length);
        }

        Ok(())
    }

    #[test]
    fn fastcdc_boundaries_match_v3_golden_fixture() -> Result<(), ChunkingError> {
        // Generated with fastcdc 3.2.1 using the same deterministic fixture.
        let mut data = Vec::with_capacity(64 * 1024 + 123);
        for i in 0usize..(64 * 1024 + 123) {
            data.push(((i * 31 + i / 7) % 251) as u8);
        }

        let descriptors = chunk_descriptors(&data, Some(1024), Some(6000), Some(16 * 1024))?;
        let boundaries: Vec<(u64, usize)> = descriptors
            .iter()
            .map(|chunk| (chunk.offset, chunk.length))
            .collect();

        assert_eq!(
            boundaries,
            vec![
                (0, 6162),
                (6162, 7028),
                (13190, 7028),
                (20218, 7028),
                (27246, 7028),
                (34274, 7028),
                (41302, 7028),
                (48330, 7028),
                (55358, 7028),
                (62386, 3273),
            ]
        );

        Ok(())
    }

    #[test]
    fn test_chunk_stream_with_hash_matches_eager() -> Result<(), ChunkingError> {
        let data = b"stream hash parity test payload".repeat(1024);
        let eager = chunk_data_with_hash(
            &data,
            Some(1024),
            Some(4096),
            Some(8192),
            HashAlgorithm::Blake3,
        )?;
        let stream = chunk_stream_with_hash(
            Cursor::new(&data),
            Some(1024),
            Some(4096),
            Some(8192),
            HashAlgorithm::Blake3,
        )?;
        assert_eq!(eager, stream);
        Ok(())
    }

    #[test]
    fn test_overflow_protection_in_chunking() -> Result<(), ChunkingError> {
        // Test that bounds checking prevents overflow
        let data = vec![0u8; 100];
        let result = chunk_data(&data, Some(1024), Some(4096), Some(4 * 1024 * 1024));
        // Should succeed with valid options
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn chunk_metadata_equality_ignores_payload() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hash as StdHash;

        let chunk_a = ChunkMetadata {
            hash: [1_u8; 32],
            offset: 10,
            length: 5,
            payload: Bytes::from_static(b"payload-a"),
        };

        let chunk_b = ChunkMetadata {
            hash: [1_u8; 32],
            offset: 10,
            length: 5,
            payload: Bytes::from_static(b"payload-b"),
        };

        assert_eq!(chunk_a, chunk_b);

        let mut hasher_a = DefaultHasher::new();
        StdHash::hash(&chunk_a, &mut hasher_a);

        let mut hasher_b = DefaultHasher::new();
        StdHash::hash(&chunk_b, &mut hasher_b);

        assert_eq!(hasher_a.finish(), hasher_b.finish());
    }

    #[test]
    fn test_max_size_validation_at_boundary() {
        let result = ChunkingOptions::resolve(
            Some(64),
            Some(1024),
            Some(1024 * 1024 * 1024), // 1GB - at the limit
        );
        assert!(result.is_ok());

        let options = result.unwrap();
        assert_eq!(options.min_size, 64);
        assert_eq!(options.avg_size, 1024);
        assert_eq!(options.max_size, 1024 * 1024 * 1024);
    }

    #[test]
    fn test_max_size_rejected_above_1gb() {
        let result = ChunkingOptions::resolve(
            Some(64),
            Some(1024),
            Some(1024 * 1024 * 1024 + 1), // 1GB + 1 byte
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ChunkingError::InvalidOptions(_)));
        let msg = err.to_string();
        assert!(
            msg.contains("1GB"),
            "Error message should mention 1GB limit: {msg}"
        );
    }

    #[test]
    fn test_error_message_clarity_for_invalid_options() {
        // Verify error messages are clear and helpful
        let result = ChunkingOptions::resolve(Some(32), Some(64), Some(128)); // min_size < 64
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("min_size"),
            "Error should mention min_size: {msg}"
        );
        assert!(
            msg.contains("64"),
            "Error should mention the limit 64: {msg}"
        );
    }

    #[test]
    fn test_valid_sizes_at_various_boundaries() {
        // Test with exact boundary values that should be valid
        let test_cases = [
            (64, 64, 64),                                   // All at minimum
            (64, 1024, 4096),                               // Typical small
            (256 * 1024, 1024 * 1024, 4 * 1024 * 1024),     // Default values
            (1024 * 1024, 1024 * 1024, 1024 * 1024 * 1024), // Max allowed
        ];

        for (min, avg, max) in test_cases {
            let result = ChunkingOptions::resolve(Some(min), Some(avg), Some(max));
            assert!(
                result.is_ok(),
                "Should accept min={min}, avg={avg}, max={max}"
            );

            let options = result.unwrap();
            assert_eq!(options.min_size, min);
            assert_eq!(options.avg_size, avg);
            assert_eq!(options.max_size, max);
        }
    }
}
