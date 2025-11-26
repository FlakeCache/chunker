use fastcdc::v2020::FastCDC;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;

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

/// Stream content-defined chunks from any reader while maintaining FastCDC state
/// across reads. Chunks are hashed via a bounded worker channel to avoid
/// unbounded buffering while preserving output order.
pub fn chunk_stream<R: Read>(
    mut reader: R,
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
) -> Result<Vec<(String, usize, usize)>, ChunkingError> {
    #[allow(clippy::cast_possible_truncation)]
    let min = min_size.unwrap_or(16_384) as u32;
    #[allow(clippy::cast_possible_truncation)]
    let avg = avg_size.unwrap_or(65_536) as u32;
    #[allow(clippy::cast_possible_truncation)]
    let max = max_size.unwrap_or(262_144) as u32;

    let mut buffer: Vec<u8> = Vec::new();
    let mut chunks: Vec<(String, usize, usize)> = Vec::new();
    let mut offset_base: usize = 0;
    let mut next_index: usize = 0;
    let mut pending_results: BTreeMap<usize, (String, usize, usize)> = BTreeMap::new();

    let (job_tx, result_rx) = crate::hashing::spawn_hashing_worker(4);
    let mut submitted = 0usize;

    let mut read_buf = [0u8; 65_536];
    loop {
        let bytes_read = reader.read(&mut read_buf)?;
        if bytes_read == 0 {
            break;
        }
        buffer.extend_from_slice(&read_buf[..bytes_read]);

        submit_ready_chunks(
            false,
            min,
            avg,
            max,
            &mut buffer,
            &mut offset_base,
            &mut next_index,
            &mut submitted,
            &job_tx,
        )?;

        drain_ready_results(
            &result_rx,
            &mut pending_results,
            &mut chunks,
            submitted,
            false,
        );
    }

    submit_ready_chunks(
        true,
        min,
        avg,
        max,
        &mut buffer,
        &mut offset_base,
        &mut next_index,
        &mut submitted,
        &job_tx,
    )?;
    drop(job_tx);

    while chunks.len() < submitted {
        drain_ready_results(
            &result_rx,
            &mut pending_results,
            &mut chunks,
            submitted,
            true,
        );
    }

    Ok(chunks)
}

fn submit_ready_chunks(
    finalize: bool,
    min: u32,
    avg: u32,
    max: u32,
    buffer: &mut Vec<u8>,
    offset_base: &mut usize,
    next_index: &mut usize,
    submitted: &mut usize,
    job_tx: &std::sync::mpsc::SyncSender<Option<crate::hashing::HashJob>>,
) -> Result<(), ChunkingError> {
    let chunker = FastCDC::new(buffer, min, avg, max);
    let mut produced: Vec<_> = chunker.collect();

    if !finalize && !produced.is_empty() {
        let _ = produced.pop();
    }

    let mut drain_up_to = 0usize;
    for chunk in produced {
        validate_slice_bounds(buffer.len(), chunk.offset, chunk.length)?;
        let start = chunk.offset;
        let end = chunk.offset + chunk.length;
        let data = buffer[start..end].to_vec();
        let global_offset = *offset_base + chunk.offset;
        let job = crate::hashing::HashJob {
            index: *next_index,
            offset: global_offset,
            data,
        };
        job_tx
            .send(Some(job))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::BrokenPipe, e))?;
        *next_index += 1;
        *submitted += 1;
        drain_up_to = end;
    }

    if drain_up_to > 0 {
        let _ = buffer.drain(..drain_up_to);
        *offset_base += drain_up_to;
    }

    Ok(())
}

fn drain_ready_results(
    result_rx: &std::sync::mpsc::Receiver<crate::hashing::HashResult>,
    pending_results: &mut BTreeMap<usize, (String, usize, usize)>,
    chunks: &mut Vec<(String, usize, usize)>,
    submitted: usize,
    block_if_empty: bool,
) {
    if submitted == chunks.len() {
        return;
    }

    flush_pending(pending_results, chunks);

    if submitted == chunks.len() {
        return;
    }

    if block_if_empty && pending_results.is_empty() {
        if let Ok(result) = result_rx.recv() {
            let _ =
                pending_results.insert(result.index, (result.digest, result.offset, result.length));
            flush_pending(pending_results, chunks);
        } else {
            return;
        }
    }

    while let Ok(result) = result_rx.try_recv() {
        let _ = pending_results.insert(result.index, (result.digest, result.offset, result.length));
        flush_pending(pending_results, chunks);
        if chunks.len() == submitted {
            break;
        }
    }
}

fn flush_pending(
    pending_results: &mut BTreeMap<usize, (String, usize, usize)>,
    chunks: &mut Vec<(String, usize, usize)>,
) {
    while let Some((index, (digest, offset, length))) = pending_results.pop_first() {
        if index == chunks.len() {
            chunks.push((digest, offset, length));
        } else {
            let _ = pending_results.insert(index, (digest, offset, length));
            break;
        }
    }
}
