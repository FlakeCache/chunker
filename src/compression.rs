use std::io::{Read, Write};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};

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

#[derive(Debug, Clone)]
pub struct CompressionJob {
    pub index: usize,
    pub payload: Vec<u8>,
    pub level: Option<i32>,
}

#[derive(Debug, Clone)]
pub struct CompressionResult {
    pub index: usize,
    pub compressed: Vec<u8>,
}

/// Spawn a bounded zstd compression worker that processes payloads in submission
/// order. The worker terminates when it receives `None`.
pub fn spawn_zstd_worker(
    bound: usize,
) -> (
    SyncSender<Option<CompressionJob>>,
    Receiver<Result<CompressionResult, CompressionError>>,
) {
    let (job_tx, job_rx) = sync_channel(bound);
    let (result_tx, result_rx) = sync_channel(bound);

    let _ = std::thread::spawn(move || {
        while let Ok(message) = job_rx.recv() {
            let Some(job): Option<CompressionJob> = message else {
                break;
            };
            let compressed = compress_zstd(&job.payload, job.level);
            let result = compressed.map(|payload| CompressionResult {
                index: job.index,
                compressed: payload,
            });
            if result_tx.send(result).is_err() {
                break;
            }
        }
    });

    (job_tx, result_rx)
}
