use rustler::{Binary, Env, Error, NewBinary, NifResult};
use std::io::{Read, Write};

/// Maximum decompressed size: 10GB
/// Protects against decompression bomb attacks where a small compressed payload
/// expands to exhaust memory (e.g., 1KB → 10GB).
const MAX_DECOMPRESSED_SIZE: u64 = 10 * 1024 * 1024 * 1024;

/// Helper function to create a binary from compressed data
fn create_binary_from_data<'a>(env: Env<'a>, data: &[u8]) -> Binary<'a> {
    let mut binary = NewBinary::new(env, data.len());
    binary.as_mut_slice().copy_from_slice(data);
    binary.into()
}

/// Compress data using `zstd`
/// Args: data (binary), `level` (optional, default 3)
#[rustler::nif]
pub fn compress_zstd<'a>(env: Env<'a>, data: Binary, level: Option<i32>) -> NifResult<Binary<'a>> {
    let compression_level = level.unwrap_or(3);

    let compressed = zstd::encode_all(data.as_slice(), compression_level)
        .map_err(|_| Error::RaiseTerm(Box::new("compression_failed")))?;

    Ok(create_binary_from_data(env, &compressed))
}

/// Decompress `zstd` data
/// Protected against decompression bombs with `MAX_DECOMPRESSED_SIZE` limit
#[rustler::nif]
pub fn decompress_zstd<'a>(env: Env<'a>, data: Binary) -> NifResult<Binary<'a>> {
    let mut decompressed = Vec::new();
    let decoder = zstd::Decoder::new(data.as_slice())
        .map_err(|e| Error::RaiseTerm(Box::new(format!("decompression_init_failed: {e}"))))?;

    // Limit reader to MAX_DECOMPRESSED_SIZE to prevent decompression bombs
    let mut limited_reader = decoder.take(MAX_DECOMPRESSED_SIZE);
    let _bytes_read = limited_reader
        .read_to_end(&mut decompressed)
        .map_err(|e| Error::RaiseTerm(Box::new(format!("decompression_failed: {e}"))))?;

    // Check if we hit the size limit (indicates potential decompression bomb)
    if decompressed.len() as u64 == MAX_DECOMPRESSED_SIZE {
        return Err(Error::RaiseTerm(Box::new("decompressed_size_exceeded")));
    }

    Ok(create_binary_from_data(env, &decompressed))
}

/// Compress data using `xz` (LZMA2)
/// Args: data (binary), `level` (optional, default 6)
#[rustler::nif]
pub fn compress_xz<'a>(env: Env<'a>, data: Binary, level: Option<u32>) -> NifResult<Binary<'a>> {
    let compression_level = level.unwrap_or(6);

    let mut encoder = xz2::write::XzEncoder::new(Vec::new(), compression_level);
    encoder
        .write_all(data.as_slice())
        .map_err(|_| Error::RaiseTerm(Box::new("compression_failed")))?;

    let compressed = encoder
        .finish()
        .map_err(|_| Error::RaiseTerm(Box::new("compression_failed")))?;

    Ok(create_binary_from_data(env, &compressed))
}

/// Decompress `xz` data
/// Protected against decompression bombs with `MAX_DECOMPRESSED_SIZE` limit
#[rustler::nif]
pub fn decompress_xz<'a>(env: Env<'a>, data: Binary) -> NifResult<Binary<'a>> {
    let decoder = xz2::read::XzDecoder::new(data.as_slice());
    let mut decompressed = Vec::new();

    // Limit reader to MAX_DECOMPRESSED_SIZE to prevent decompression bombs
    let mut limited_reader = decoder.take(MAX_DECOMPRESSED_SIZE);
    let _bytes_read = limited_reader
        .read_to_end(&mut decompressed)
        .map_err(|e| Error::RaiseTerm(Box::new(format!("decompression_failed: {e}"))))?;

    // Check if we hit the size limit (indicates potential decompression bomb)
    if decompressed.len() as u64 == MAX_DECOMPRESSED_SIZE {
        return Err(Error::RaiseTerm(Box::new("decompressed_size_exceeded")));
    }

    Ok(create_binary_from_data(env, &decompressed))
}
