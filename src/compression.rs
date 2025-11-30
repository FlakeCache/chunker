use std::io::{Read, Write};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use tracing::{debug, instrument, warn};

/// Scratch buffers that can be reused to reduce allocations in tight loops.
#[derive(Debug, Default)]
pub struct CompressionScratch {
    pub output: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum CompressionError {
    #[error("compression_failed: {0}")]
    Compression(String),
    #[error("decompression_failed: {0}")]
    Decompression(String),
    #[error("decompressed_size_exceeded")]
    SizeExceeded,
    #[error("unknown_format")]
    UnknownFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionStrategy {
    Fastest,  // LZ4
    Balanced, // Zstd
    Smallest, // XZ
}

/// Compress data using the best algorithm for the given strategy.
///
/// # Errors
///
/// Returns `CompressionError` if compression fails.
#[instrument(skip(data), fields(data_len = data.len(), strategy = ?strategy))]
pub fn compress(data: &[u8], strategy: CompressionStrategy) -> Result<Vec<u8>, CompressionError> {
    match strategy {
        CompressionStrategy::Fastest => compress_lz4(data),
        CompressionStrategy::Balanced => compress_zstd(data, None),
        CompressionStrategy::Smallest => compress_xz(data, None),
    }
}

/// Automatically detect compression format and decompress into a provided buffer.
/// Supports: Zstd, LZ4 (prepend-size), XZ, Bzip2
///
/// # Errors
///
/// Returns `CompressionError` if decompression fails or format is unknown.
pub fn decompress_auto_into(data: &[u8], output: &mut Vec<u8>) -> Result<(), CompressionError> {
    if data.len() < 4 {
        return Err(CompressionError::UnknownFormat);
    }

    // Magic bytes checks

    // Zstd: 0xFD2FB528 (Little Endian) -> 28 B5 2F FD
    if data.starts_with(&[0x28, 0xB5, 0x2F, 0xFD]) {
        debug!("detected_format_zstd");
        return decompress_zstd_into(data, output);
    }

    // XZ: FD 37 7A 58 5A 00
    if data.starts_with(&[0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00]) {
        debug!("detected_format_xz");
        return decompress_xz_into(data, output);
    }

    // Bzip2: BZh
    if data.starts_with(b"BZh") {
        debug!("detected_format_bzip2");
        return decompress_bzip2_into(data, output);
    }

    // LZ4 Frame: 04 22 4D 18 (Little Endian)
    if data.starts_with(&[0x04, 0x22, 0x4D, 0x18]) {
        debug!("detected_format_lz4");
        return decompress_lz4_into(data, output);
    }

    Err(CompressionError::UnknownFormat)
}

/// Automatically detect compression format and decompress.
/// Supports: Zstd, LZ4 (prepend-size), XZ, Bzip2
///
/// # Errors
///
/// Returns `CompressionError` if decompression fails or format is unknown.
#[instrument(skip(data), fields(data_len = data.len()))]
pub fn decompress_auto(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
    let mut decompressed = Vec::new();
    decompress_auto_into(data, &mut decompressed)?;
    Ok(decompressed)
}

/// A reader that automatically detects compression format and transparently decompresses.
pub struct AutoDecompressReader<R: Read + Send> {
    inner: Box<dyn Read + Send>,
    _marker: std::marker::PhantomData<R>,
}

impl<R: Read + Send> std::fmt::Debug for AutoDecompressReader<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutoDecompressReader")
            .field("inner", &"Box<dyn Read + Send>")
            .finish()
    }
}

impl<R: Read + Send + 'static> AutoDecompressReader<R> {
    /// Create a new auto-decompressing reader.
    /// This will buffer the first few bytes to detect the format.
    ///
    /// # Errors
    ///
    /// Returns `std::io::Error` if reading from the stream fails.
    pub fn new(mut reader: R) -> std::io::Result<Self> {
        let mut header = [0u8; 6];
        let n = reader.read(&mut header)?;
        let header_slice = &header[..n];

        // Reconstruct the stream: header + rest
        let stream = std::io::Cursor::new(header_slice.to_vec()).chain(reader);

        // Detect format
        if header_slice.starts_with(&[0x28, 0xB5, 0x2F, 0xFD]) {
            debug!("detected_format_zstd_stream");
            let decoder = zstd::Decoder::new(stream)?;
            return Ok(Self {
                inner: Box::new(decoder.take(MAX_DECOMPRESSED_SIZE)),
                _marker: std::marker::PhantomData,
            });
        }

        // XZ: FD 37 7A 58 5A 00
        if header_slice.starts_with(&[0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00]) {
            debug!("detected_format_xz_stream");
            // TODO: lzma-rs does not support streaming read yet.
            // let decoder = xz2::read::XzDecoder::new(stream);
            // return Ok(Self {
            //     inner: Box::new(decoder.take(MAX_DECOMPRESSED_SIZE)),
            //     _marker: std::marker::PhantomData,
            // });
            warn!("xz_stream_decompression_not_supported_with_lzma_rs");
        }

        if header_slice.starts_with(b"BZh") {
            debug!("detected_format_bzip2_stream");
            let decoder = bzip2::read::BzDecoder::new(stream);
            return Ok(Self {
                inner: Box::new(decoder.take(MAX_DECOMPRESSED_SIZE)),
                _marker: std::marker::PhantomData,
            });
        }

        if header_slice.starts_with(&[0x04, 0x22, 0x4D, 0x18]) {
            debug!("detected_format_lz4_stream");
            let decoder = lz4_flex::frame::FrameDecoder::new(stream);
            return Ok(Self {
                inner: Box::new(decoder.take(MAX_DECOMPRESSED_SIZE)),
                _marker: std::marker::PhantomData,
            });
        }

        // Fallback: Assume uncompressed
        debug!("detected_format_uncompressed_stream");
        Ok(Self {
            inner: Box::new(stream.take(MAX_DECOMPRESSED_SIZE)),
            _marker: std::marker::PhantomData,
        })
    }
}

impl<R: Read + Send> Read for AutoDecompressReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buf)
    }
}

/// Maximum decompressed size: 1 GB
/// Protects against decompression bomb attacks where a small compressed payload
/// expands to exhaust memory (e.g., 1KB → >1 GB).
const DEFAULT_MAX_DECOMPRESSED_SIZE: u64 = 1024 * 1024 * 1024; // 1 GB
const MAX_DECOMPRESSED_SIZE: u64 = DEFAULT_MAX_DECOMPRESSED_SIZE;

/// Decompress `zstd` data into a provided buffer
///
/// # Errors
///
/// Returns `CompressionError` if decompression fails or size limit is exceeded.
pub fn decompress_zstd_into(data: &[u8], output: &mut Vec<u8>) -> Result<(), CompressionError> {
    decompress_zstd_into_with_limit(data, output, DEFAULT_MAX_DECOMPRESSED_SIZE)
}

/// Decompress `zstd` data into a provided buffer with a configurable size limit
///
/// # Errors
///
/// Returns `CompressionError` if decompression fails or size limit is exceeded.
pub fn decompress_zstd_into_with_limit(
    data: &[u8],
    output: &mut Vec<u8>,
    limit: u64,
) -> Result<(), CompressionError> {
    decompress_zstd_into_with_limit_and_dict(data, output, limit, None)
}

/// Decompress `zstd` data into a provided buffer with a configurable size limit and dictionary
///
/// # Errors
///
/// Returns `CompressionError` if decompression fails or size limit is exceeded.
pub fn decompress_zstd_into_with_limit_and_dict(
    data: &[u8],
    output: &mut Vec<u8>,
    limit: u64,
    dict: Option<&zstd::dict::DecoderDictionary>,
) -> Result<(), CompressionError> {
    if let Some(d) = dict {
        let reader = std::io::BufReader::new(std::io::Cursor::new(data));
        let decoder = zstd::Decoder::with_prepared_dictionary(reader, d)
            .map_err(|e| CompressionError::Decompression(format!("init: {e}")))?;

        let mut limited_reader = decoder.take(limit + 1);
        let start_len = output.len();
        let _bytes_read = limited_reader
            .read_to_end(output)
            .map_err(|e| CompressionError::Decompression(e.to_string()))?;

        if (output.len() - start_len) as u64 > limit {
            warn!("zstd_decompression_bomb_detected");
            return Err(CompressionError::SizeExceeded);
        }
        debug!(
            decompressed_len = output.len() - start_len,
            "zstd_decompression_complete"
        );
        return Ok(());
    }

    let reader = std::io::BufReader::new(std::io::Cursor::new(data));
    let decoder = zstd::Decoder::new(reader)
        .map_err(|e| CompressionError::Decompression(format!("init: {e}")))?;

    let mut limited_reader = decoder.take(limit + 1);
    let start_len = output.len();
    let _bytes_read = limited_reader
        .read_to_end(output)
        .map_err(|e| CompressionError::Decompression(e.to_string()))?;

    if (output.len() - start_len) as u64 > limit {
        warn!("zstd_decompression_bomb_detected");
        return Err(CompressionError::SizeExceeded);
    }
    debug!(
        decompressed_len = output.len() - start_len,
        "zstd_decompression_complete"
    );
    Ok(())
}

/// Compress data using `lz4` (Frame format)
/// Args: data (binary)
///
/// # Errors
///
/// Returns `CompressionError` if compression fails.
#[instrument(skip(data), fields(data_len = data.len()))]
pub fn compress_lz4(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
    let mut encoder = lz4_flex::frame::FrameEncoder::new(Vec::with_capacity(data.len() / 2));
    encoder
        .write_all(data)
        .map_err(|e| CompressionError::Compression(e.to_string()))?;
    let compressed = encoder
        .finish()
        .map_err(|e| CompressionError::Compression(e.to_string()))?;

    debug!(
        compressed_len = compressed.len(),
        "lz4_compression_complete"
    );
    Ok(compressed)
}

/// Compress data using `lz4` into a provided buffer (cleared before use).
///
/// # Errors
///
/// Returns `CompressionError` if compression fails.
pub fn compress_lz4_into(data: &[u8], output: &mut Vec<u8>) -> Result<(), CompressionError> {
    output.clear();
    let mut encoder = lz4_flex::frame::FrameEncoder::new(std::mem::take(output));
    encoder
        .write_all(data)
        .map_err(|e| CompressionError::Compression(e.to_string()))?;
    let compressed = encoder
        .finish()
        .map_err(|e| CompressionError::Compression(e.to_string()))?;
    *output = compressed;
    Ok(())
}

/// Decompress `lz4` data (Frame format) into a provided buffer.
///
/// # Errors
///
/// Returns `CompressionError` if decompression fails or size limit is exceeded.
pub fn decompress_lz4_into(data: &[u8], output: &mut Vec<u8>) -> Result<(), CompressionError> {
    let decoder = lz4_flex::frame::FrameDecoder::new(data);

    // Limit reader to MAX_DECOMPRESSED_SIZE + 1 to detect overflow
    let mut limited_reader = decoder.take(MAX_DECOMPRESSED_SIZE + 1);
    let start_len = output.len();
    let _bytes_read = limited_reader
        .read_to_end(output)
        .map_err(|e| CompressionError::Decompression(e.to_string()))?;

    // Check if we hit the size limit
    if (output.len() - start_len) as u64 > MAX_DECOMPRESSED_SIZE {
        warn!("lz4_decompression_bomb_detected");
        return Err(CompressionError::SizeExceeded);
    }

    debug!(
        decompressed_len = output.len() - start_len,
        "lz4_decompression_complete"
    );
    Ok(())
}

/// Decompress `lz4` data (Frame format)
///
/// # Errors
///
/// Returns `CompressionError` if decompression fails or size limit is exceeded.
#[instrument(skip(data), fields(data_len = data.len()))]
pub fn decompress_lz4(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
    let mut decompressed = Vec::new();
    decompress_lz4_into(data, &mut decompressed)?;
    Ok(decompressed)
}

/// Compress data using `zstd`
/// Args: data (binary), `level` (optional, default 3)
///
/// # Errors
///
/// Returns `CompressionError` if compression fails.
#[instrument(skip(data), fields(data_len = data.len(), level = ?level))]
pub fn compress_zstd(data: &[u8], level: Option<i32>) -> Result<Vec<u8>, CompressionError> {
    let mut compressed = Vec::with_capacity(data.len() / 4 + 1024);
    compress_zstd_into_internal(data, level, None, &mut compressed)?;
    Ok(compressed)
}

/// Compress data using `zstd` into a provided buffer (cleared before use).
///
/// # Errors
///
/// Returns `CompressionError` if compression fails.
#[instrument(skip(data, output, dict), fields(data_len = data.len(), level = ?level))]
pub fn compress_zstd_into(
    data: &[u8],
    level: Option<i32>,
    dict: Option<&zstd::dict::EncoderDictionary>,
    output: &mut Vec<u8>,
) -> Result<(), CompressionError> {
    output.clear();
    compress_zstd_into_internal(data, level, dict, output)
}

fn compress_zstd_into_internal(
    data: &[u8],
    level: Option<i32>,
    dict: Option<&zstd::dict::EncoderDictionary>,
    output: &mut Vec<u8>,
) -> Result<(), CompressionError> {
    let compression_level = level.unwrap_or(3);

    let mut encoder = match dict {
        // Prepared dictionaries do not take an explicit level; the dictionary already encodes parameters.
        Some(d) => zstd::Encoder::with_prepared_dictionary(&mut *output, d),
        None => zstd::Encoder::new(&mut *output, compression_level),
    }
    .map_err(|e| CompressionError::Compression(e.to_string()))?;

    encoder
        .write_all(data)
        .map_err(|e| CompressionError::Compression(e.to_string()))?;
    let _ = encoder
        .finish()
        .map_err(|e| CompressionError::Compression(e.to_string()))?;

    debug!(compressed_len = output.len(), "zstd_compression_complete");
    Ok(())
}

/// Decompress `zstd` data using an optional dictionary into a provided buffer.
///
/// # Errors
///
/// Returns `CompressionError` if decompression fails or size limit is exceeded.
pub fn decompress_zstd_with_dict_into(
    data: &[u8],
    output: &mut Vec<u8>,
    dict: Option<&zstd::dict::DecoderDictionary>,
) -> Result<(), CompressionError> {
    decompress_zstd_into_with_limit_and_dict(data, output, MAX_DECOMPRESSED_SIZE, dict)
}

/// Decompress `zstd` data
/// Protected against decompression bombs with `DEFAULT_MAX_DECOMPRESSED_SIZE` limit
///
/// # Errors
///
/// Returns `CompressionError` if decompression fails or size limit is exceeded.
#[instrument(skip(data), fields(data_len = data.len()))]
pub fn decompress_zstd(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
    let mut decompressed = Vec::new();
    decompress_zstd_into(data, &mut decompressed)?;
    Ok(decompressed)
}

/// Decompress `zstd` data with a configurable size limit
///
/// # Errors
///
/// Returns `CompressionError` if decompression fails or size limit is exceeded.
#[instrument(skip(data), fields(data_len = data.len(), limit = limit))]
pub fn decompress_zstd_with_limit(data: &[u8], limit: u64) -> Result<Vec<u8>, CompressionError> {
    let mut decompressed = Vec::new();
    decompress_zstd_into_with_limit(data, &mut decompressed, limit)?;
    Ok(decompressed)
}

/// Compress data using `xz` (LZMA2)
/// Args: data (binary), `level` (optional, default 6)
///
/// # Errors
///
/// Returns `CompressionError` if compression fails.
pub fn compress_xz(data: &[u8], _level: Option<u32>) -> Result<Vec<u8>, CompressionError> {
    // lzma-rs defaults to level 6 equivalent
    let mut output = Vec::with_capacity(data.len() / 2);
    let mut input = data;
    lzma_rs::xz_compress(&mut input, &mut output)
        .map_err(|e| CompressionError::Compression(e.to_string()))?;

    Ok(output)
}

/// Decompress `xz` data into a provided buffer.
/// Protected against decompression bombs with `MAX_DECOMPRESSED_SIZE` limit
///
/// # Errors
///
/// Returns `CompressionError` if decompression fails or size limit is exceeded.
pub fn decompress_xz_into(data: &[u8], output: &mut Vec<u8>) -> Result<(), CompressionError> {
    struct LimitedWriter<'a> {
        inner: &'a mut Vec<u8>,
        limit: u64,
        written: u64,
    }

    impl Write for LimitedWriter<'_> {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if self.written + buf.len() as u64 > self.limit {
                return Err(std::io::Error::other("SizeExceeded"));
            }
            let n = self.inner.write(buf)?;
            self.written += n as u64;
            Ok(n)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            self.inner.flush()
        }
    }

    let mut input = data;
    let mut writer = LimitedWriter {
        inner: output,
        limit: MAX_DECOMPRESSED_SIZE,
        written: 0,
    };

    lzma_rs::xz_decompress(&mut input, &mut writer).map_err(|e| {
        let s = e.to_string();
        if s.contains("SizeExceeded") {
            CompressionError::SizeExceeded
        } else {
            CompressionError::Decompression(s)
        }
    })?;

    Ok(())
}

/// Decompress `xz` data
/// Protected against decompression bombs with `MAX_DECOMPRESSED_SIZE` limit
///
/// # Errors
///
/// Returns `CompressionError` if decompression fails or size limit is exceeded.
pub fn decompress_xz(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
    let mut decompressed = Vec::new();
    decompress_xz_into(data, &mut decompressed)?;
    Ok(decompressed)
}

/// Compress data using `bzip2`
/// Args: data (binary), `level` (optional, default 6)
///
/// # Errors
///
/// Returns `CompressionError` if compression fails.
pub fn compress_bzip2(data: &[u8], level: Option<u32>) -> Result<Vec<u8>, CompressionError> {
    let compression_level = level.unwrap_or(6);
    let compression = bzip2::Compression::new(compression_level);

    let mut encoder = bzip2::write::BzEncoder::new(Vec::with_capacity(data.len() / 2), compression);
    encoder
        .write_all(data)
        .map_err(|e| CompressionError::Compression(e.to_string()))?;

    let compressed = encoder
        .finish()
        .map_err(|e| CompressionError::Compression(e.to_string()))?;

    Ok(compressed)
}

/// Decompress `bzip2` data into a provided buffer.
/// Protected against decompression bombs with `MAX_DECOMPRESSED_SIZE` limit
///
/// # Errors
///
/// Returns `CompressionError` if decompression fails or size limit is exceeded.
pub fn decompress_bzip2_into(data: &[u8], output: &mut Vec<u8>) -> Result<(), CompressionError> {
    let decoder = bzip2::read::BzDecoder::new(data);

    // Limit reader to MAX_DECOMPRESSED_SIZE to prevent decompression bombs
    let mut limited_reader = decoder.take(MAX_DECOMPRESSED_SIZE);
    let start_len = output.len();
    let _bytes_read = limited_reader
        .read_to_end(output)
        .map_err(|e| CompressionError::Decompression(e.to_string()))?;

    // Check if we hit the size limit (indicates potential decompression bomb)
    if (output.len() - start_len) as u64 == MAX_DECOMPRESSED_SIZE {
        return Err(CompressionError::SizeExceeded);
    }

    Ok(())
}

/// Decompress `bzip2` data
/// Protected against decompression bombs with `MAX_DECOMPRESSED_SIZE` limit
///
/// # Errors
///
/// Returns `CompressionError` if decompression fails or size limit is exceeded.
pub fn decompress_bzip2(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
    let mut decompressed = Vec::new();
    decompress_bzip2_into(data, &mut decompressed)?;
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

/// Handle for a Zstd compression worker.
pub type ZstdWorkerHandle = (
    SyncSender<Option<CompressionJob>>,
    Receiver<Result<CompressionResult, CompressionError>>,
    std::thread::JoinHandle<()>,
);

/// Spawn a bounded zstd compression worker that processes payloads in submission
/// order.
///
/// The worker terminates when it receives `None`.
/// Returns (sender, receiver, `worker_handle`) for panic detection and synchronization.
#[must_use]
pub fn spawn_zstd_worker(bound: usize) -> ZstdWorkerHandle {
    let (job_tx, job_rx) = sync_channel(bound);
    let (result_tx, result_rx) = sync_channel(bound);

    let handle = std::thread::spawn(move || {
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

    (job_tx, result_rx, handle)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn test_xz_roundtrip() -> Result<(), CompressionError> {
        let data = b"hello world xz compression test";
        let compressed = compress_xz(data, None)?;
        let decompressed = decompress_xz(&compressed)?;
        assert_eq!(data, decompressed.as_slice());
        Ok(())
    }

    #[test]
    fn test_bzip2_roundtrip() -> Result<(), CompressionError> {
        let data = b"hello world bzip2 compression test";
        let compressed = compress_bzip2(data, None)?;
        let decompressed = decompress_bzip2(&compressed)?;
        assert_eq!(data, decompressed.as_slice());
        Ok(())
    }

    #[test]
    fn test_lz4_roundtrip() -> Result<(), CompressionError> {
        let data = b"hello world lz4 compression test";
        let compressed = compress_lz4(data)?;
        let decompressed = decompress_lz4(&compressed)?;
        assert_eq!(data, decompressed.as_slice());
        Ok(())
    }

    #[test]
    fn test_lz4_into_reuses_buffer() -> Result<(), CompressionError> {
        let data = b"buffer reuse test payload";
        let mut buf = vec![1, 2, 3];
        compress_lz4_into(data, &mut buf)?;
        let decompressed = decompress_lz4(&buf)?;
        assert_eq!(data, decompressed.as_slice());
        Ok(())
    }

    #[test]
    fn test_auto_decompression() -> Result<(), CompressionError> {
        let data = b"auto decompression test payload";

        // Test Zstd
        let zstd_compressed = compress_zstd(data, None)?;
        assert_eq!(decompress_auto(&zstd_compressed)?, data);

        // Test LZ4
        let lz4_compressed = compress_lz4(data)?;
        assert_eq!(decompress_auto(&lz4_compressed)?, data);

        // Test XZ
        let xz_compressed = compress_xz(data, None)?;
        assert_eq!(decompress_auto(&xz_compressed)?, data);

        // Test Bzip2
        let bzip2_compressed = compress_bzip2(data, None)?;
        assert_eq!(decompress_auto(&bzip2_compressed)?, data);

        Ok(())
    }

    #[test]
    fn test_zstd_dict_roundtrip() -> Result<(), CompressionError> {
        let data = b"dictionary test payload data";
        let dict_bytes = b"tiny-zstd-dictionary";

        let enc_dict = zstd::dict::EncoderDictionary::copy(dict_bytes, 3);
        let dec_dict = zstd::dict::DecoderDictionary::copy(dict_bytes);

        let mut compressed = Vec::new();
        compress_zstd_into(data, Some(3), Some(&enc_dict), &mut compressed)?;

        let mut decompressed = Vec::new();
        decompress_zstd_with_dict_into(&compressed, &mut decompressed, Some(&dec_dict))?;
        assert_eq!(decompressed.as_slice(), data);
        Ok(())
    }

    #[test]
    fn test_compression_strategy_smallest_hits_xz() -> Result<(), CompressionError> {
        let data = b"strategy test payload for xz";
        let compressed = compress(data, CompressionStrategy::Smallest)?;
        let mut output = Vec::new();
        decompress_auto_into(&compressed, &mut output)?;
        assert_eq!(output.as_slice(), data);
        Ok(())
    }

    #[test]
    fn test_compression_strategy_fastest_lz4_roundtrip() -> Result<(), CompressionError> {
        let data = b"strategy fastest payload";
        let compressed = compress(data, CompressionStrategy::Fastest)?;
        let mut output = Vec::new();
        decompress_auto_into(&compressed, &mut output)?;
        assert_eq!(output.as_slice(), data);
        Ok(())
    }

    #[test]
    fn test_compression_strategy_balanced_zstd_roundtrip() -> Result<(), CompressionError> {
        let data = b"strategy balanced payload";
        let compressed = compress(data, CompressionStrategy::Balanced)?;
        let mut output = Vec::new();
        decompress_auto_into(&compressed, &mut output)?;
        assert_eq!(output.as_slice(), data);
        Ok(())
    }

    #[test]
    fn test_decompress_auto_unknown_format() {
        let data = [0u8; 4]; // fails all magic checks
        let mut output = Vec::new();
        let result = decompress_auto_into(&data, &mut output);
        assert!(matches!(result, Err(CompressionError::UnknownFormat)));
    }

    #[test]
    fn test_auto_decompress_reader() -> Result<(), std::io::Error> {
        let data = b"streaming auto decompression test payload";

        // Test Zstd Stream
        let zstd_compressed = compress_zstd(data, None).unwrap();
        let mut reader = AutoDecompressReader::new(std::io::Cursor::new(zstd_compressed))?;
        let mut buffer = Vec::new();
        let _ = reader.read_to_end(&mut buffer)?;
        assert_eq!(buffer, data);

        // Test LZ4 Stream
        let lz4_compressed = compress_lz4(data).unwrap();
        let mut reader = AutoDecompressReader::new(std::io::Cursor::new(lz4_compressed))?;
        let mut buffer = Vec::new();
        let _ = reader.read_to_end(&mut buffer)?;
        assert_eq!(buffer, data);

        // Test Uncompressed Fallback
        let mut reader = AutoDecompressReader::new(std::io::Cursor::new(data))?;
        let mut buffer = Vec::new();
        let _ = reader.read_to_end(&mut buffer)?;
        assert_eq!(buffer, data);

        Ok(())
    }

    #[test]
    fn test_zstd_worker() -> Result<(), String> {
        let (tx, rx, handle) = spawn_zstd_worker(10);

        let data = b"worker test data";
        tx.send(Some(CompressionJob {
            index: 0,
            payload: data.to_vec(),
            level: None,
        }))
        .map_err(|err| err.to_string())?;
        tx.send(None).map_err(|err| err.to_string())?;

        let result = rx
            .recv()
            .map_err(|err| err.to_string())?
            .map_err(|err| err.to_string())?;
        assert_eq!(result.index, 0);

        let decompressed = decompress_zstd(&result.compressed).map_err(|err| err.to_string())?;
        assert_eq!(decompressed, data);

        handle.join().map_err(|_| "worker panicked".to_string())?;
        Ok(())
    }
}
