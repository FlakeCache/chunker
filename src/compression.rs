use std::io::{Read, Write};

#[derive(Debug, thiserror::Error)]
pub enum CompressionError {
    #[error("compression_failed: {0}")]
    Compression(String),
    #[error("decompression_failed: {0}")]
    Decompression(String),
    #[error("decompressed_size_exceeded")]
    SizeExceeded,
}

/// Maximum decompressed size: 10GB
/// Protects against decompression bomb attacks where a small compressed payload
/// expands to exhaust memory (e.g., 1KB → 10GB).
const MAX_DECOMPRESSED_SIZE: u64 = 10 * 1024 * 1024 * 1024;

/// Compress data using `zstd`
/// Args: data (binary), `level` (optional, default 3)
pub fn compress_zstd(data: &[u8], level: Option<i32>) -> Result<Vec<u8>, CompressionError> {
    let compression_level = level.unwrap_or(3);

    let compressed = zstd::encode_all(data, compression_level)
        .map_err(|e| CompressionError::Compression(e.to_string()))?;

    Ok(compressed)
}

/// Decompress `zstd` data
/// Protected against decompression bombs with `MAX_DECOMPRESSED_SIZE` limit
pub fn decompress_zstd(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
    let mut decompressed = Vec::new();
    let decoder = zstd::Decoder::new(data)
        .map_err(|e| CompressionError::Decompression(format!("init: {e}")))?;

    // Limit reader to MAX_DECOMPRESSED_SIZE to prevent decompression bombs
    let mut limited_reader = decoder.take(MAX_DECOMPRESSED_SIZE);
    let _bytes_read = limited_reader
        .read_to_end(&mut decompressed)
        .map_err(|e| CompressionError::Decompression(e.to_string()))?;

    // Check if we hit the size limit (indicates potential decompression bomb)
    if decompressed.len() as u64 == MAX_DECOMPRESSED_SIZE {
        return Err(CompressionError::SizeExceeded);
    }

    Ok(decompressed)
}

/// Compress data using `xz` (LZMA2)
/// Args: data (binary), `level` (optional, default 6)
pub fn compress_xz(data: &[u8], level: Option<u32>) -> Result<Vec<u8>, CompressionError> {
    let compression_level = level.unwrap_or(6);

    let mut encoder = xz2::write::XzEncoder::new(Vec::new(), compression_level);
    encoder
        .write_all(data)
        .map_err(|e| CompressionError::Compression(e.to_string()))?;

    let compressed = encoder
        .finish()
        .map_err(|e| CompressionError::Compression(e.to_string()))?;

    Ok(compressed)
}

/// Decompress `xz` data
/// Protected against decompression bombs with `MAX_DECOMPRESSED_SIZE` limit
pub fn decompress_xz(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
    let decoder = xz2::read::XzDecoder::new(data);
    let mut decompressed = Vec::new();

    // Limit reader to MAX_DECOMPRESSED_SIZE to prevent decompression bombs
    let mut limited_reader = decoder.take(MAX_DECOMPRESSED_SIZE);
    let _bytes_read = limited_reader
        .read_to_end(&mut decompressed)
        .map_err(|e| CompressionError::Decompression(e.to_string()))?;

    // Check if we hit the size limit (indicates potential decompression bomb)
    if decompressed.len() as u64 == MAX_DECOMPRESSED_SIZE {
        return Err(CompressionError::SizeExceeded);
    }

    Ok(decompressed)
}

/// Compress data using `bzip2`
/// Args: data (binary), `level` (optional, default 6)
pub fn compress_bzip2(data: &[u8], level: Option<u32>) -> Result<Vec<u8>, CompressionError> {
    let compression_level = level.unwrap_or(6);

    let mut encoder =
        bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::new(compression_level));
    encoder
        .write_all(data)
        .map_err(|e| CompressionError::Compression(e.to_string()))?;

    let compressed = encoder
        .finish()
        .map_err(|e| CompressionError::Compression(e.to_string()))?;

    Ok(compressed)
}

/// Decompress `bzip2` data
/// Protected against decompression bombs with `MAX_DECOMPRESSED_SIZE` limit
pub fn decompress_bzip2(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
    let decoder = bzip2::read::BzDecoder::new(data);
    let mut decompressed = Vec::new();

    // Limit reader to MAX_DECOMPRESSED_SIZE to prevent decompression bombs
    let mut limited_reader = decoder.take(MAX_DECOMPRESSED_SIZE);
    let _bytes_read = limited_reader
        .read_to_end(&mut decompressed)
        .map_err(|e| CompressionError::Decompression(e.to_string()))?;

    // Check if we hit the size limit (indicates potential decompression bomb)
    if decompressed.len() as u64 == MAX_DECOMPRESSED_SIZE {
        return Err(CompressionError::SizeExceeded);
    }

    Ok(decompressed)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &[u8] = b"Chunker compression round trip data for testing";

    #[test]
    fn zstd_round_trip() {
        let compressed = compress_zstd(SAMPLE, Some(3)).expect("zstd compress");
        let decompressed = decompress_zstd(&compressed).expect("zstd decompress");
        assert_eq!(decompressed, SAMPLE);
    }

    #[test]
    fn xz_round_trip() {
        let compressed = compress_xz(SAMPLE, Some(6)).expect("xz compress");
        let decompressed = decompress_xz(&compressed).expect("xz decompress");
        assert_eq!(decompressed, SAMPLE);
    }

    #[test]
    fn bzip2_round_trip() {
        let compressed = compress_bzip2(SAMPLE, Some(6)).expect("bzip2 compress");
        let decompressed = decompress_bzip2(&compressed).expect("bzip2 decompress");
        assert_eq!(decompressed, SAMPLE);
    }
}
