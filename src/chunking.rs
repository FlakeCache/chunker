use fastcdc::v2020::{FastCDC, StreamCDC};
use sha2::{Digest, Sha256};

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
    #[error("offset_overflow: offset {offset} cannot fit into usize")]
    OffsetOverflow { offset: u64 },
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
pub fn chunk_data(
    data: &[u8],
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
) -> Result<Vec<(String, usize, usize)>, ChunkingError> {
    // These values are well below u32::MAX, so truncation is safe
    #[allow(clippy::cast_possible_truncation)]
    let min = min_size.unwrap_or(16_384) as u32; // 16 KB
    #[allow(clippy::cast_possible_truncation)]
    let avg = avg_size.unwrap_or(65_536) as u32; // 64 KB
    #[allow(clippy::cast_possible_truncation)]
    let max = max_size.unwrap_or(262_144) as u32; // 256 KB

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

    Ok(chunks)
}

/// Chunk streaming data using `FastCDC` without materializing the full payload in memory.
///
/// The chunk boundaries will match those produced by [`chunk_data`], and each chunk's SHA256
/// hash is computed incrementally from the streamed data.
pub fn chunk_stream<R: std::io::Read>(
    reader: R,
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
) -> Result<Vec<(String, usize, usize)>, ChunkingError> {
    #[allow(clippy::cast_possible_truncation)]
    let min = min_size.unwrap_or(16_384) as u32; // 16 KB
    #[allow(clippy::cast_possible_truncation)]
    let avg = avg_size.unwrap_or(65_536) as u32; // 64 KB
    #[allow(clippy::cast_possible_truncation)]
    let max = max_size.unwrap_or(262_144) as u32; // 256 KB

    let chunker = StreamCDC::new(reader, min, avg, max);
    let mut chunks = Vec::new();

    for chunk_result in chunker {
        let chunk = chunk_result.map_err(std::io::Error::from)?;

        let chunk_offset =
            usize::try_from(chunk.offset).map_err(|_| ChunkingError::OffsetOverflow {
                offset: chunk.offset,
            })?;

        let mut hasher = Sha256::new();
        hasher.update(&chunk.data);
        let hash_hex = hex::encode(hasher.finalize());

        chunks.push((hash_hex, chunk_offset, chunk.length));
    }

    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn stream_and_buffer_chunking_match() {
        let data: Vec<u8> = (0u8..=255).cycle().take(10_000).collect();

        let buffered = chunk_data(&data, Some(256), Some(1024), Some(4_096)).unwrap();
        let streamed = chunk_stream(
            Cursor::new(data.as_slice()),
            Some(256),
            Some(1024),
            Some(4_096),
        )
        .unwrap();

        assert_eq!(buffered, streamed);
    }
}
