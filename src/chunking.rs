use fastcdc::v2020::FastCDC;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressorPreset {
    Zstd { level: i32 },
    Xz { level: u32 },
}

impl CompressorPreset {
    pub const fn codec(&self) -> &'static str {
        match self {
            Self::Zstd { .. } => "zstd",
            Self::Xz { .. } => "xz",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolicyProfile {
    pub name: &'static str,
    pub min_size: usize,
    pub avg_size: usize,
    pub max_size: usize,
    pub compressor: CompressorPreset,
}

impl PolicyProfile {
    /// General-purpose profile balanced for throughput and deduplication.
    pub const fn balanced() -> Self {
        Self {
            name: "balanced",
            min_size: 16_384,
            avg_size: 65_536,
            max_size: 262_144,
            compressor: CompressorPreset::Zstd { level: 3 },
        }
    }

    /// Favor higher throughput with fewer, larger chunks and faster compression.
    pub const fn throughput_optimized() -> Self {
        Self {
            name: "throughput_optimized",
            min_size: 32_768,
            avg_size: 131_072,
            max_size: 524_288,
            compressor: CompressorPreset::Zstd { level: 1 },
        }
    }

    /// Favor deduplication density and maximum compression ratio.
    pub const fn archival() -> Self {
        Self {
            name: "archival",
            min_size: 8_192,
            avg_size: 32_768,
            max_size: 131_072,
            compressor: CompressorPreset::Xz { level: 6 },
        }
    }

    pub const fn chunking_bounds(&self) -> (usize, usize, usize) {
        (self.min_size, self.avg_size, self.max_size)
    }
}

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

#[cfg(test)]
mod tests {
    use super::{CompressorPreset, PolicyProfile};

    #[test]
    fn policy_profile_balanced_matches_defaults() {
        let profile = PolicyProfile::balanced();
        assert_eq!(profile.name, "balanced");
        assert_eq!(profile.chunking_bounds(), (16_384, 65_536, 262_144));
        assert_eq!(profile.compressor, CompressorPreset::Zstd { level: 3 });
        assert_eq!(profile.compressor.codec(), "zstd");
    }

    #[test]
    fn policy_profile_throughput_prefers_larger_chunks() {
        let profile = PolicyProfile::throughput_optimized();
        assert_eq!(profile.name, "throughput_optimized");
        assert_eq!(profile.chunking_bounds(), (32_768, 131_072, 524_288));
        assert_eq!(profile.compressor, CompressorPreset::Zstd { level: 1 });
    }

    #[test]
    fn policy_profile_archival_prefers_density() {
        let profile = PolicyProfile::archival();
        assert_eq!(profile.name, "archival");
        assert_eq!(profile.chunking_bounds(), (8_192, 32_768, 131_072));
        assert_eq!(profile.compressor, CompressorPreset::Xz { level: 6 });
        assert_eq!(profile.compressor.codec(), "xz");
    }
}
