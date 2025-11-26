use fastcdc::v2020::FastCDC;
use sha2::{Digest, Sha256};

#[cfg(feature = "quickcdc")]
use quickcdc::Chunker as QuickChunker;

#[derive(Debug, thiserror::Error, Clone, Copy)]
pub enum ChunkingError {
    #[error(
        "bounds_check_failed: offset {offset} + length {length} exceeds data length {data_len}"
    )]
    Bounds {
        data_len: usize,
        offset: usize,
        length: usize,
    },
    #[error("invalid_chunker_parameters: {reason}")]
    InvalidParameters { reason: &'static str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkingStrategyKind {
    FastCdc,
    #[cfg(feature = "quickcdc")]
    QuickCdc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkBoundary {
    pub offset: usize,
    pub length: usize,
}

pub trait ChunkerStrategy {
    fn chunk(
        &self,
        data: &[u8],
        min_size: usize,
        avg_size: usize,
        max_size: usize,
    ) -> Result<Vec<ChunkBoundary>, ChunkingError>;
}

#[derive(Debug, Default)]
struct FastCdcStrategy;

impl ChunkerStrategy for FastCdcStrategy {
    fn chunk(
        &self,
        data: &[u8],
        min_size: usize,
        avg_size: usize,
        max_size: usize,
    ) -> Result<Vec<ChunkBoundary>, ChunkingError> {
        // These values are well below u32::MAX, so truncation is safe
        #[allow(clippy::cast_possible_truncation)]
        let min = min_size as u32;
        #[allow(clippy::cast_possible_truncation)]
        let avg = avg_size as u32;
        #[allow(clippy::cast_possible_truncation)]
        let max = max_size as u32;

        let chunker = FastCDC::new(data, min, avg, max);

        let mut chunks = Vec::new();

        for chunk in chunker {
            // Validate bounds before slice access (defense-in-depth)
            validate_slice_bounds(data.len(), chunk.offset, chunk.length)?;

            chunks.push(ChunkBoundary {
                offset: chunk.offset,
                length: chunk.length,
            });
        }

        Ok(chunks)
    }
}

#[cfg(feature = "quickcdc")]
#[derive(Debug)]
struct QuickCdcStrategy {
    salt: u64,
}

#[cfg(feature = "quickcdc")]
impl Default for QuickCdcStrategy {
    fn default() -> Self {
        Self {
            salt: 0x9e3779b97f4a7c15,
        }
    }
}

#[cfg(feature = "quickcdc")]
impl ChunkerStrategy for QuickCdcStrategy {
    fn chunk(
        &self,
        data: &[u8],
        min_size: usize,
        avg_size: usize,
        max_size: usize,
    ) -> Result<Vec<ChunkBoundary>, ChunkingError> {
        if avg_size < 64 {
            return Err(ChunkingError::InvalidParameters {
                reason: "average chunk size must be >= 64 bytes for quickcdc",
            });
        }

        let mut chunker =
            QuickChunker::with_params(data, avg_size, max_size, self.salt).map_err(|_| {
                ChunkingError::InvalidParameters {
                    reason: "quickcdc failed to initialize with provided sizes",
                }
            })?;

        let mut chunks = Vec::new();
        let mut offset = 0usize;
        let mut pending: Option<ChunkBoundary> = None;

        while let Some(slice) = chunker.next() {
            let mut boundary = ChunkBoundary {
                offset,
                length: slice.len(),
            };

            if let Some(pending_chunk) = pending.take() {
                boundary.offset = pending_chunk.offset;
                boundary.length += pending_chunk.length;
            }

            while boundary.length > max_size {
                chunks.push(ChunkBoundary {
                    offset: boundary.offset,
                    length: max_size,
                });
                boundary.offset += max_size;
                boundary.length -= max_size;
            }

            if boundary.length == 0 {
                offset += slice.len();
                continue;
            }

            if boundary.length < min_size {
                pending = Some(boundary);
            } else {
                chunks.push(boundary);
            }

            offset += slice.len();
        }

        if let Some(pending_chunk) = pending {
            chunks.push(pending_chunk);
        }

        Ok(chunks)
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

fn resolve_sizes(
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
) -> Result<(usize, usize, usize), ChunkingError> {
    let min = min_size.unwrap_or(16_384); // 16 KB
    let avg = avg_size.unwrap_or(65_536); // 64 KB
    let max = max_size.unwrap_or(262_144); // 256 KB

    if min == 0 || avg == 0 || max == 0 {
        return Err(ChunkingError::InvalidParameters {
            reason: "chunk sizes must be non-zero",
        });
    }

    if min > max || avg > max {
        return Err(ChunkingError::InvalidParameters {
            reason: "min/avg chunk sizes must not exceed max size",
        });
    }

    Ok((min, avg, max))
}

pub fn chunk_boundaries_with_strategy(
    data: &[u8],
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
    strategy: ChunkingStrategyKind,
) -> Result<Vec<ChunkBoundary>, ChunkingError> {
    let (min, avg, max) = resolve_sizes(min_size, avg_size, max_size)?;

    match strategy {
        ChunkingStrategyKind::FastCdc => FastCdcStrategy.chunk(data, min, avg, max),
        #[cfg(feature = "quickcdc")]
        ChunkingStrategyKind::QuickCdc => QuickCdcStrategy::default().chunk(data, min, avg, max),
    }
}

/// Chunk data using the selected content-defined chunking strategy.
/// Args: data (binary), `min_size` (optional), `avg_size` (optional), `max_size` (optional), `strategy`
/// Returns: list of {`chunk_hash`, `offset`, `length`}
pub fn chunk_data_with_strategy(
    data: &[u8],
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
    strategy: ChunkingStrategyKind,
) -> Result<Vec<(String, usize, usize)>, ChunkingError> {
    let chunks = chunk_boundaries_with_strategy(data, min_size, avg_size, max_size, strategy)?;

    let mut results = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        validate_slice_bounds(data.len(), chunk.offset, chunk.length)?;

        // Compute SHA256 hash of chunk
        let mut hasher = Sha256::new();
        hasher.update(&data[chunk.offset..chunk.offset + chunk.length]);
        let hash = hasher.finalize();
        let hash_hex = hex::encode(hash);

        results.push((hash_hex, chunk.offset, chunk.length));
    }

    Ok(results)
}

/// Backwards-compatible default: chunk using FastCDC
pub fn chunk_data(
    data: &[u8],
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
) -> Result<Vec<(String, usize, usize)>, ChunkingError> {
    chunk_data_with_strategy(
        data,
        min_size,
        avg_size,
        max_size,
        ChunkingStrategyKind::FastCdc,
    )
}
