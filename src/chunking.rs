#![allow(clippy::cast_precision_loss)]
use blake3;
use bytes::{Bytes, BytesMut};
use fastcdc::v2020::FastCDC;
use metrics::{counter, histogram};
#[cfg(feature = "async-stream")]
use async_stream::try_stream;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::io::{ErrorKind, Read};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, instrument, trace};
#[cfg(feature = "async-stream")]
use futures::io::AsyncRead;
#[cfg(feature = "async-stream")]
use futures::io::AsyncReadExt;
#[cfg(feature = "async-stream")]
use futures::executor::block_on;
#[cfg(feature = "async-stream")]
use futures::stream::Stream;
#[cfg(feature = "async-stream")]
use futures::stream::StreamExt;
#[cfg(feature = "async-stream")]
use std::pin::Pin;
#[cfg(feature = "async-stream")]
use std::task::{Context, Poll};

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
}

/// Hash algorithm used when producing chunk metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HashAlgorithm {
    Blake3,
    Sha256,
}

/// Metadata for a single chunk emitted by streaming chunkers.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// SHA-256 or Blake3 hash of the chunk payload (32 bytes).
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
    pub fn resolve(min_size: Option<usize>, avg_size: Option<usize>, max_size: Option<usize>) -> Result<Self, ChunkingError> {
        let options = Self {
            min_size: min_size.unwrap_or(256 * 1024),       // 256 KB
            avg_size: avg_size.unwrap_or(1024 * 1024),      // 1 MB
            max_size: max_size.unwrap_or(4 * 1024 * 1024),  // 4 MB
        };
        
        options.validate()?;
        
        trace!(?options, "resolved_chunking_options");
        Ok(options)
    }

    fn validate(&self) -> Result<(), ChunkingError> {
        if self.min_size < 64 {
            return Err(ChunkingError::InvalidOptions("min_size must be >= 64".into()));
        }
        if self.min_size > self.avg_size {
            return Err(ChunkingError::InvalidOptions("min_size must be <= avg_size".into()));
        }
        if self.avg_size > self.max_size {
            return Err(ChunkingError::InvalidOptions("avg_size must be <= max_size".into()));
        }
        if self.max_size > 1024 * 1024 * 1024 {
            return Err(ChunkingError::InvalidOptions("max_size must be <= 1GB".into()));
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
    let options = ChunkingOptions::resolve(min_size, avg_size, max_size)?;

    // 1. FastCDC Pass (Serial, Memory-Bound, ~2.5 GB/s)
    // We collect cut points first to enable parallel hashing.
    let chunker = FastCDC::new(
        data,
        options.min_size as u32,
        options.avg_size as u32,
        options.max_size as u32,
    );

    let cut_points: Vec<_> = chunker.collect();

    // 2. Hashing Pass (Parallel, CPU-Bound)
    // Rayon will distribute the hashing of chunks across all available cores.
    let chunks: Result<Vec<ChunkMetadata>, ChunkingError> = cut_points
        .par_iter()
        .map(|chunk_def| {
            let offset = chunk_def.offset;
            let length = chunk_def.length;

            // Safety check to prevent panics in the thread pool
            if offset + length > data.len() {
                return Err(ChunkingError::Bounds {
                    data_len: data.len(),
                    offset,
                    length,
                });
            }

            let chunk_slice = &data[offset..offset + length];

            let hash_array: [u8; 32] = match hash {
                HashAlgorithm::Sha256 => {
                    let mut hasher = Sha256::new();
                    hasher.update(chunk_slice);
                    hasher.finalize().into()
                }
                HashAlgorithm::Blake3 => blake3::hash(chunk_slice).into(),
            };

            Ok(ChunkMetadata {
                hash: hash_array,
                offset: offset as u64,
                length,
                payload: Bytes::copy_from_slice(chunk_slice),
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
    avg_size: usize,
    max_size: usize,
    hash: HashAlgorithm,
    position: u64,
    eof: bool,
}

static TRACE_SAMPLE_COUNTER: AtomicU64 = AtomicU64::new(0);
const TRACE_SAMPLE_EVERY: u64 = 1024;
#[cfg(feature = "async-stream")]
const ASYNC_BUFFER_LIMIT: usize = 512 * 1024 * 1024; // 512 MiB safety cap

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
    pub fn new(reader: R, min_size: Option<usize>, avg_size: Option<usize>, max_size: Option<usize>) -> Result<Self, ChunkingError> {
        Self::new_with_hash(reader, min_size, avg_size, max_size, HashAlgorithm::Sha256)
    }

    /// Create a streaming chunker with an explicit hash algorithm.
    ///
    /// This initializes the internal slab buffer lazily to avoid large allocations upfront.
    ///
    /// # Errors
    ///
    /// Returns `ChunkingError::InvalidOptions` if the provided sizes are invalid.
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
        })
    }
}

impl<R: Read> Iterator for ChunkStream<R> {
    type Item = Result<ChunkMetadata, ChunkingError>;

    #[allow(clippy::too_many_lines)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // 1. Try to find a chunk in the current buffer
            if !self.buffer.is_empty() {
                // Create a FastCDC iterator over the buffer
                let mut cdc = FastCDC::new(
                    &self.buffer,
                    self.min_size as u32,
                    self.avg_size as u32,
                    self.max_size as u32,
                );

                if let Some(chunk) = cdc.next() {
                    // Found a chunk!
                    let len = chunk.length;
                    let offset = chunk.offset;

                    // Safety check: FastCDC should never return a zero-length chunk
                    if len == 0 {
                        return Some(Err(ChunkingError::ZeroLengthChunk));
                    }

                    // Safety check: FastCDC should start at 0 for the first chunk in the slice
                    if offset != 0 {
                        return Some(Err(std::io::Error::new(
                            ErrorKind::InvalidData,
                            format!("FastCDC returned non-zero offset {offset} for first chunk"),
                        ).into()));
                    }

                    // Safety check: Ensure we don't panic on split_to
                    if len > self.buffer.len() {
                        return Some(Err(ChunkingError::Bounds {
                            data_len: self.buffer.len(),
                            offset: 0,
                            length: len,
                        }));
                    }

                    // If FastCDC consumed the whole buffer but we're not at EOF and haven't reached max_size,
                    // it means we likely haven't found a true cut point yet. We should read more data.
                    if len != self.buffer.len() || self.eof || len >= self.max_size {
                        let chunk_data = self.buffer.split_to(len).freeze();

                        // Metrics
                        counter!("chunker.chunks_emitted").increment(1);
                        counter!("chunker.bytes_processed").increment(len as u64);
                        histogram!("chunker.chunk_size").record(len as f64);
                        
                        let hash_array: [u8; 32] = match self.hash {
                            HashAlgorithm::Sha256 => {
                                let mut hasher = Sha256::new();
                                hasher.update(&chunk_data);
                                hasher.finalize().into()
                            }
                            HashAlgorithm::Blake3 => blake3::hash(&chunk_data).into(),
                        };

                        let chunk_offset = self.position;
                        self.position += len as u64;

                        if tracing::enabled!(tracing::Level::TRACE)
                            && TRACE_SAMPLE_COUNTER.fetch_add(1, Ordering::Relaxed).is_multiple_of(TRACE_SAMPLE_EVERY)
                        {
                            trace!(offset = chunk_offset, length = len, "chunk_emitted");
                        }

                        return Some(Ok(ChunkMetadata {
                            hash: hash_array,
                            offset: chunk_offset,
                            length: len,
                            payload: chunk_data,
                        }));
                    }
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

                // Metrics
                counter!("chunker.chunks_emitted").increment(1);
                counter!("chunker.bytes_processed").increment(len as u64);
                histogram!("chunker.chunk_size").record(len as f64);
                
                let hash_array: [u8; 32] = match self.hash {
                    HashAlgorithm::Sha256 => {
                        let mut hasher = Sha256::new();
                        hasher.update(&chunk_data);
                        hasher.finalize().into()
                    }
                    HashAlgorithm::Blake3 => blake3::hash(&chunk_data).into(),
                };

                let chunk_offset = self.position;
                self.position += len as u64;
                
                return Some(Ok(ChunkMetadata {
                    hash: hash_array,
                    offset: chunk_offset,
                    length: len,
                    payload: chunk_data,
                }));
            }

            // 3. Read more data
            // We want to read a decent amount.
            let read_size = std::cmp::max(self.max_size, 4096);
            
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
        f.debug_struct("ChunkStreamAsync")
            .finish_non_exhaustive()
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
    pub fn new(reader: R, min_size: Option<usize>, avg_size: Option<usize>, max_size: Option<usize>) -> Result<Self, ChunkingError> {
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
                        options.min_size as u32,
                        options.avg_size as u32,
                        options.max_size as u32,
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

                            counter!("chunker.chunks_emitted").increment(1);
                            counter!("chunker.bytes_processed").increment(len as u64);
                            histogram!("chunker.chunk_size").record(len as f64);
                            
                            let hash_array: [u8; 32] = match hash {
                                HashAlgorithm::Sha256 => {
                                    let mut hasher = Sha256::new();
                                    hasher.update(&chunk_data);
                                    hasher.finalize().into()
                                }
                                HashAlgorithm::Blake3 => blake3::hash(&chunk_data).into(),
                            };

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

                    counter!("chunker.chunks_emitted").increment(1);
                    counter!("chunker.bytes_processed").increment(len as u64);
                    histogram!("chunker.chunk_size").record(len as f64);
                    
                    let hash_array: [u8; 32] = match hash {
                        HashAlgorithm::Sha256 => {
                            let mut hasher = Sha256::new();
                            hasher.update(&chunk_data);
                            hasher.finalize().into()
                        }
                        HashAlgorithm::Blake3 => blake3::hash(&chunk_data).into(),
                    };

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

                // 3. Read more data
                let read_size = std::cmp::max(options.max_size, 4096);
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
/// This function enforces `ASYNC_BUFFER_LIMIT` on the total accumulated payload size.
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
    let mut stream = ChunkStreamAsync::new(reader, min_size, avg_size, max_size)?;
    let mut chunks = Vec::new();
    let mut total_len = 0;

    while let Some(chunk_res) = stream.next().await {
        let chunk = chunk_res?;
        total_len += chunk.length;
        
        if total_len > ASYNC_BUFFER_LIMIT {
            return Err(ChunkingError::Io(std::io::Error::new(
                ErrorKind::OutOfMemory,
                "chunk_data_async buffer limit exceeded",
            )));
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
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};

    #[test]
    fn streaming_reader_emits_full_length() -> Result<(), ChunkingError> {
        let size = 3 * 1024 * 1024 + 321; // Just over 3 MiB
        let data = vec![42_u8; size];
        let cursor = Cursor::new(&data);
        let reader = BufReader::new(cursor);

        let mut stream = ChunkStream::new(reader, Some(16 * 1024), Some(32 * 1024), Some(64 * 1024))?;
        let mut total = 0_usize;
        let mut count = 0_usize;

        while let Some(chunk) = stream.next() {
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
        let collected: Vec<_> = streaming
            .by_ref()
            .collect::<Result<_, _>>()?;

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
    fn test_chunk_stream_with_hash_matches_eager() -> Result<(), ChunkingError> {
        let data = b"stream hash parity test payload".repeat(1024);
        let eager = chunk_data_with_hash(&data, Some(1024), Some(4096), Some(8192), HashAlgorithm::Blake3)?;
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
}
