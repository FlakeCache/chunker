use fastcdc::v2020::FastCDC;
use rustler::{Binary, NifResult};
use sha2::{Digest, Sha256};

/// Validate slice bounds to prevent out-of-bounds access
/// Returns an error if offset + length would exceed `data_len` or overflow
fn validate_slice_bounds(data_len: usize, offset: usize, length: usize) -> NifResult<()> {
    if offset.checked_add(length).is_none_or(|end| end > data_len) {
        return Err(rustler::Error::RaiseTerm(Box::new(format!(
            "bounds_check_failed: offset {offset} + length {length} exceeds data length {data_len}"
        ))));
    }
    Ok(())
}

/// Chunk data using `FastCDC` (Content-Defined Chunking)
/// Args: data (binary), `min_size` (optional), `avg_size` (optional), `max_size` (optional)
/// Returns: list of {`chunk_hash`, `offset`, `length`}
#[rustler::nif]
#[allow(clippy::unnecessary_wraps)] // NIFs require Result wrapper for error handling
pub fn chunk_data(
    data: Binary,
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
) -> NifResult<Vec<(String, usize, usize)>> {
    // These values are well below u32::MAX, so truncation is safe
    #[allow(clippy::cast_possible_truncation)]
    let min = min_size.unwrap_or(16_384) as u32; // 16 KB
    #[allow(clippy::cast_possible_truncation)]
    let avg = avg_size.unwrap_or(65_536) as u32; // 64 KB
    #[allow(clippy::cast_possible_truncation)]
    let max = max_size.unwrap_or(262_144) as u32; // 256 KB

    let chunker = FastCDC::new(data.as_slice(), min, avg, max);

    let mut chunks = Vec::new();

    for chunk in chunker {
        // Validate bounds before slice access (defense-in-depth)
        validate_slice_bounds(data.len(), chunk.offset, chunk.length)?;

        // Compute SHA256 hash of chunk
        let mut hasher = Sha256::new();
        hasher.update(&data.as_slice()[chunk.offset..chunk.offset + chunk.length]);
        let hash = hasher.finalize();
        let hash_hex = hex::encode(hash);

        chunks.push((hash_hex, chunk.offset, chunk.length));
    }

    Ok(chunks)
}
