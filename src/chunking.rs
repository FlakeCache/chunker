use fastcdc::v2020::{FastCDC, StreamCDC};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::io::Read;
use tracing::{debug, instrument, trace};

#[cfg(feature = "async-stream")]
use futures::io::AsyncRead;
#[cfg(feature = "async-stream")]
use futures::io::AsyncReadExt;

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
}

/// Metadata for a single chunk emitted by streaming chunkers.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Hex-encoded SHA-256 hash of the chunk payload.
    pub hash: String,
    /// Starting byte offset of the chunk relative to the reader.
    pub offset: u64,
    /// Chunk length in bytes.
    pub length: usize,
}

/// Configurable bounds for FastCDC chunking.
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
    #[instrument(level = "debug", skip(min_size, avg_size, max_size), fields(min = ?min_size, avg = ?avg_size, max = ?max_size))]
    fn resolve(min_size: Option<usize>, avg_size: Option<usize>, max_size: Option<usize>) -> Self {
        let options = Self {
            min_size: min_size.unwrap_or(16_384),
            avg_size: avg_size.unwrap_or(65_536),
            max_size: max_size.unwrap_or(262_144),
        };
        trace!(?options, "resolved_chunking_options");
        options
    }
}

/// Validate slice bounds to prevent out-of-bounds access
/// Returns an error if offset + length would exceed `data_len` or overflow
fn validate_slice_bounds(
    data_len: usize,
    offset: usize,
    length: usize,
) -> Result<(), ChunkingError> {
    if offset.checked_add(length).is_none_or(|end| end > data_len) {
        return Err(ChunkingError::Bounds {
            data_len,
            offset,
            length,
        });
    }
    Ok(())
}

/// Chunk data using `FastCDC` (Content-Defined Chunking)
/// Args: data (binary), `min_size` (optional), `avg_size` (optional), `max_size` (optional)
/// Returns: list of {`chunk_hash`, `offset`, `length`}
#[instrument(skip(data), fields(data_len = data.len()))]
pub fn chunk_data(
    data: &[u8],
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
) -> Result<Vec<(String, usize, usize)>, ChunkingError> {
    let options = ChunkingOptions::resolve(min_size, avg_size, max_size);

    // These values are well below u32::MAX, so truncation is safe
    #[allow(clippy::cast_possible_truncation)]
    let min = options.min_size as u32; // 16 KB
    #[allow(clippy::cast_possible_truncation)]
    let avg = options.avg_size as u32; // 64 KB
    #[allow(clippy::cast_possible_truncation)]
    let max = options.max_size as u32; // 256 KB

    let chunker = FastCDC::new(data, min, avg, max);

    let mut chunks = Vec::new();

    for chunk in chunker {
        // Validate bounds before slice access (defense-in-depth)
        validate_slice_bounds(data.len(), chunk.offset, chunk.length)?;

        // Compute SHA256 hash of chunk
        let mut hasher = Sha256::new();
        hasher.update(&data[chunk.offset..chunk.offset + chunk.length]);
        let hash = hasher.finalize();
        let hash_hex = hex::encode(hash);

        chunks.push((hash_hex, chunk.offset, chunk.length));
    }

    debug!(chunk_count = chunks.len(), "chunking_complete");
    Ok(chunks)
}

/// A streaming chunk reader that yields chunk metadata and hashes without buffering
/// the entire source in memory.
pub struct ChunkStream<R: Read> {
    inner: StreamCDC<R>,
}

impl<R: Read> fmt::Debug for ChunkStream<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChunkStream")
            .field("offset", &"streaming")
            .finish()
    }
}

impl<R: Read> ChunkStream<R> {
    /// Create a streaming chunker around any [`std::io::Read`] implementation.
    /// Uses FastCDC with optional min/avg/max overrides.
    #[instrument(skip(reader))]
    pub fn new(reader: R, min_size: Option<usize>, avg_size: Option<usize>, max_size: Option<usize>) -> Self {
        let options = ChunkingOptions::resolve(min_size, avg_size, max_size);

        #[allow(clippy::cast_possible_truncation)]
        let min = options.min_size as u32;
        #[allow(clippy::cast_possible_truncation)]
        let avg = options.avg_size as u32;
        #[allow(clippy::cast_possible_truncation)]
        let max = options.max_size as u32;

        let inner = StreamCDC::new(reader, min, avg, max);
        Self { inner }
    }
}

impl<R: Read> Iterator for ChunkStream<R> {
    type Item = Result<ChunkMetadata, ChunkingError>;

    fn next(&mut self) -> Option<Self::Item> {
        let chunk_result = self.inner.next()?;
        match chunk_result {
            Ok(chunk) => {
                let mut hasher = Sha256::new();
                hasher.update(&chunk.data);
                let hash_hex = hex::encode(hasher.finalize());

                trace!(hash = ?hash_hex, offset = chunk.offset, length = chunk.length, "chunk_emitted");

                Some(Ok(ChunkMetadata {
                    hash: hash_hex,
                    offset: chunk.offset,
                    length: chunk.length,
                }))
            }
            Err(err) => {
                debug!(error = ?err, "chunking_error");
                Some(Err(ChunkingError::Io(err.into())))
            },
        }
    }
}

/// Adapter to allow [`ChunkStream`] usage with asynchronous readers.
#[cfg(feature = "async-stream")]
pub struct AsyncReadAdapter<R: AsyncRead + Unpin> {
    inner: R,
}

#[cfg(feature = "async-stream")]
impl<R: AsyncRead + Unpin> fmt::Debug for AsyncReadAdapter<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AsyncReadAdapter").finish()
    }
}

#[cfg(feature = "async-stream")]
impl<R: AsyncRead + Unpin> Read for AsyncReadAdapter<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        futures::executor::block_on(self.inner.read(buf))
    }
}

/// A convenience alias that exposes streaming chunking for any [`AsyncRead`]
/// implementor under the `async-stream` feature flag.
#[cfg(feature = "async-stream")]
pub type AsyncChunkStream<R> = ChunkStream<AsyncReadAdapter<R>>;

/// Construct a [`ChunkStream`] around an asynchronous reader by blocking on
/// individual reads. This keeps the same low-memory chunking behavior while
/// allowing consumers in async contexts to feed data into FastCDC.
#[cfg(feature = "async-stream")]
pub fn chunk_stream_async<R: AsyncRead + Unpin>(
    reader: R,
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
) -> AsyncChunkStream<R> {
    ChunkStream::new(AsyncReadAdapter { inner: reader }, min_size, avg_size, max_size)
}

/// Convenience function to chunk a stream and collect all chunk metadata into a vector.
/// This is useful when you want to process a stream but need all chunk metadata at once.
/// Note: The offset is cast to `usize` to match `chunk_data`'s return type.
pub fn chunk_stream<R: Read>(
    reader: R,
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
) -> Result<Vec<(String, usize, usize)>, ChunkingError> {
    let stream = ChunkStream::new(reader, min_size, avg_size, max_size);
    let mut chunks = Vec::new();
    for chunk in stream {
        let chunk = chunk?;
        #[allow(clippy::cast_possible_truncation)]
        let offset = chunk.offset as usize;
        chunks.push((chunk.hash, offset, chunk.length));
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

        let mut stream = ChunkStream::new(reader, None, Some(32 * 1024), Some(64 * 1024));
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
        let mut streaming = ChunkStream::new(Cursor::new(&payload), None, None, None);
        let collected: Vec<_> = streaming
            .by_ref()
            .map(|res| res.map(|chunk| (chunk.hash, chunk.offset as usize, chunk.length)))
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
        for (_hash, offset, length) in &chunks {
            assert_eq!(*offset, total_len);
            total_len += length;
        }
        assert_eq!(total_len, data.len());
        Ok(())
    }

    #[test]
    fn test_chunk_data_small() -> Result<(), ChunkingError> {
        let data = b"small data";
        let chunks = chunk_data(data, None, None, None)?;

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].1, 0);
        assert_eq!(chunks[0].2, data.len());
        Ok(())
    }
}

#[cfg(test)]
mod internal_tests {
    use super::*;

    #[test]
    fn test_validate_slice_bounds() {
        assert!(validate_slice_bounds(100, 0, 100).is_ok());
        assert!(validate_slice_bounds(100, 10, 90).is_ok());
        
        // Out of bounds
        assert!(matches!(validate_slice_bounds(100, 0, 101), Err(ChunkingError::Bounds { .. })));
        assert!(matches!(validate_slice_bounds(100, 90, 20), Err(ChunkingError::Bounds { .. })));
        
        // Overflow
        assert!(matches!(validate_slice_bounds(100, usize::MAX, 1), Err(ChunkingError::Bounds { .. })));
    }
}
