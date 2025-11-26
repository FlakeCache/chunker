use fastcdc::v2020::FastCDC;
use sha2::{Digest, Sha256};

use crate::compression::CompressionSettings;
use crate::manifest::{CdcParameters, ChunkRecord, Manifest, ManifestError, ManifestFormat};

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
    #[error("manifest_serialization_failed: {0}")]
    Manifest(&'static str),
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
    let (chunks, _manifest) = chunk_data_with_manifest(
        data,
        min_size,
        avg_size,
        max_size,
        CompressionSettings::default_zstd(),
    )?;

    Ok(chunks)
}

pub fn chunk_data_with_manifest(
    data: &[u8],
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
    compression: CompressionSettings,
) -> Result<(Vec<(String, usize, usize)>, Manifest), ChunkingError> {
    // These values are well below u32::MAX, so truncation is safe
    #[allow(clippy::cast_possible_truncation)]
    let min = min_size.unwrap_or(16_384) as u32; // 16 KB
    #[allow(clippy::cast_possible_truncation)]
    let avg = avg_size.unwrap_or(65_536) as u32; // 64 KB
    #[allow(clippy::cast_possible_truncation)]
    let max = max_size.unwrap_or(262_144) as u32; // 256 KB

    let chunker = FastCDC::new(data, min, avg, max);

    let mut chunks = Vec::new();
    let mut manifest_chunks = Vec::new();

    for chunk in chunker {
        // Validate bounds before slice access (defense-in-depth)
        validate_slice_bounds(data.len(), chunk.offset, chunk.length)?;

        // Compute SHA256 hash of chunk
        let mut hasher = Sha256::new();
        hasher.update(&data[chunk.offset..chunk.offset + chunk.length]);
        let hash = hasher.finalize();
        let hash_hex = hex::encode(hash);

        let offset_u64 =
            u64::try_from(chunk.offset).map_err(|_| ChunkingError::Manifest("offset_overflow"))?;
        let length_u64 =
            u64::try_from(chunk.length).map_err(|_| ChunkingError::Manifest("length_overflow"))?;

        chunks.push((hash_hex.clone(), chunk.offset, chunk.length));
        manifest_chunks.push(ChunkRecord {
            hash: hash_hex,
            offset: offset_u64,
            length: length_u64,
        });
    }

    let manifest = Manifest {
        chunks: manifest_chunks,
        cdc: CdcParameters {
            strategy: "fastcdc".to_string(),
            min_size: u64::from(min),
            avg_size: u64::from(avg),
            max_size: u64::from(max),
        },
        compression: compression.into(),
    };

    Ok((chunks, manifest))
}

pub fn chunk_data_with_manifest_bytes(
    data: &[u8],
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
    compression: CompressionSettings,
    format: ManifestFormat,
) -> Result<(Vec<(String, usize, usize)>, Vec<u8>), ChunkingError> {
    let (chunks, manifest) =
        chunk_data_with_manifest(data, min_size, avg_size, max_size, compression)?;

    let manifest_bytes = manifest
        .to_bytes(format)
        .map_err(|manifest_error| map_manifest_error(&manifest_error))?;

    Ok((chunks, manifest_bytes))
}

fn map_manifest_error(error: &ManifestError) -> ChunkingError {
    match error {
        ManifestError::Json(_) => ChunkingError::Manifest("json"),
        ManifestError::Cbor(_) => ChunkingError::Manifest("cbor"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_is_deterministic_for_identical_inputs() {
        let data = b"deterministic manifest test data";

        let (_, manifest_bytes_first) = chunk_data_with_manifest_bytes(
            data,
            None,
            None,
            None,
            CompressionSettings::default_zstd(),
            ManifestFormat::Json,
        )
        .expect("first manifest generation should succeed");

        let (_, manifest_bytes_second) = chunk_data_with_manifest_bytes(
            data,
            None,
            None,
            None,
            CompressionSettings::default_zstd(),
            ManifestFormat::Json,
        )
        .expect("second manifest generation should succeed");

        assert_eq!(manifest_bytes_first, manifest_bytes_second);
    }

    #[test]
    fn manifest_cbor_is_deterministic_for_identical_inputs() {
        let data = b"deterministic manifest test data";

        let (_, manifest_bytes_first) = chunk_data_with_manifest_bytes(
            data,
            None,
            None,
            None,
            CompressionSettings::default_zstd(),
            ManifestFormat::Cbor,
        )
        .expect("first manifest generation should succeed");

        let (_, manifest_bytes_second) = chunk_data_with_manifest_bytes(
            data,
            None,
            None,
            None,
            CompressionSettings::default_zstd(),
            ManifestFormat::Cbor,
        )
        .expect("second manifest generation should succeed");

        assert_eq!(manifest_bytes_first, manifest_bytes_second);
    }
}
