use fastcdc::v2020::FastCDC;
use sha2::{Digest, Sha256};

use crate::manifest::{self, ChunkDescriptor, CdcStrategy, CompressionSettings, HashAlgorithm, Manifest};

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
}

#[derive(Debug, Clone, Copy)]
pub struct CdcConfig {
    pub min_size: u32,
    pub avg_size: u32,
    pub max_size: u32,
}

impl Default for CdcConfig {
    fn default() -> Self {
        Self {
            min_size: 16_384,
            avg_size: 65_536,
            max_size: 262_144,
        }
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
pub fn chunk_data(
    data: &[u8],
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
) -> Result<Vec<(String, usize, usize)>, ChunkingError> {
    let config = CdcConfig {
        min_size: min_size.unwrap_or(16_384) as u32,
        avg_size: avg_size.unwrap_or(65_536) as u32,
        max_size: max_size.unwrap_or(262_144) as u32,
    };

    chunk_data_with_manifest(data, config, CompressionSettings::default())
        .map(|manifest| {
            manifest
                .chunks
                .iter()
                .map(|chunk| {
                    (
                        chunk.hash.clone(),
                        chunk.offset as usize,
                        chunk.length as usize,
                    )
                })
                .collect()
        })
}

pub fn chunk_data_with_manifest(
    data: &[u8],
    config: CdcConfig,
    compression: CompressionSettings,
) -> Result<Manifest, ChunkingError> {
    let chunker = FastCDC::new(data, config.min_size, config.avg_size, config.max_size);

    let mut chunks: Vec<ChunkDescriptor> = Vec::new();
    let mut leaf_hashes: Vec<Vec<u8>> = Vec::new();

    for chunk in chunker {
        validate_slice_bounds(data.len(), chunk.offset, chunk.length)?;

        let mut hasher = Sha256::new();
        hasher.update(&data[chunk.offset..chunk.offset + chunk.length]);
        let hash_bytes = hasher.finalize().to_vec();
        let hash_hex = hex::encode(&hash_bytes);

        leaf_hashes.push(hash_bytes);

        chunks.push(ChunkDescriptor {
            offset: chunk.offset as u64,
            length: chunk.length as u64,
            hash: hash_hex,
        });
    }

    let merkle_root = manifest::compute_merkle_root(&leaf_hashes)
        .map(hex::encode)
        .unwrap_or_default();

    Ok(Manifest {
        version: 1,
        hash_algorithm: HashAlgorithm::Sha256,
        cdc_strategy: CdcStrategy::FastCdc {
            min: config.min_size,
            avg: config.avg_size,
            max: config.max_size,
        },
        compression,
        merkle_root,
        chunks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merkle_root_present_in_manifest() {
        let data = b"abcdefghijklmnopqrstuvwxyz";
        let manifest = chunk_data_with_manifest(
            data,
            CdcConfig::default(),
            CompressionSettings::default(),
        )
        .expect("manifest generation");

        assert!(!manifest.chunks.is_empty());
        assert!(!manifest.merkle_root.is_empty());

        let json = manifest
            .to_canonical_json_bytes()
            .expect("json serialization");
        let decoded = Manifest::from_json_bytes(&json).expect("json parse");
        assert_eq!(manifest.merkle_root, decoded.merkle_root);
    }
}
