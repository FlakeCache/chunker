use fastcdc::v2020::FastCDC;
use crossbeam_channel::bounded;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::thread;

use crate::{compression, hashing};

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
    #[error("compression_error: {0}")]
    Compression(#[from] compression::CompressionError),
    #[error("thread_panic: {0}")]
    ThreadPanic(String),
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

#[derive(Debug, Clone)]
pub struct StreamChunk {
    pub index: usize,
    pub offset: usize,
    pub length: usize,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct ProcessedChunk {
    pub index: usize,
    pub offset: usize,
    pub length: usize,
    pub hash: String,
    pub compressed: Vec<u8>,
}

/// Stream chunks from a reader, maintaining FastCDC state across incremental reads
pub fn chunk_stream<R: Read + Send + 'static>(
    reader: R,
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
) -> Result<Vec<(String, usize, usize)>, ChunkingError> {
    let mut chunks = Vec::new();

    stream_chunks(reader, min_size, avg_size, max_size, |chunk| {
        let hash = hashing::sha256_hash(&chunk.data);
        chunks.push((hash, chunk.offset, chunk.length));
        Ok(())
    })?;

    Ok(chunks)
}

/// Stream chunks through a hashing and compression pipeline with bounded channels
pub fn process_stream_with_pipeline<R: Read + Send + 'static>(
    reader: R,
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
    compression_level: Option<i32>,
    channel_capacity: usize,
) -> Result<Vec<ProcessedChunk>, ChunkingError> {
    let (chunk_tx, chunk_rx) = bounded::<StreamChunk>(channel_capacity);
    let (hashed_tx, hashed_rx) = bounded::<(StreamChunk, String)>(channel_capacity);
    let (compressed_tx, compressed_rx) =
        bounded::<Result<ProcessedChunk, compression::CompressionError>>(channel_capacity);

    // Hashing worker
    let hashing_handle = thread::spawn(move || {
        for chunk in chunk_rx {
            let hash = hashing::sha256_hash(&chunk.data);
            if hashed_tx.send((chunk, hash)).is_err() {
                break;
            }
        }
    });

    // Compression worker
    let compression_handle = thread::spawn(move || {
        for (chunk, hash) in hashed_rx {
            let compressed = match compression::compress_zstd(&chunk.data, compression_level) {
                Ok(result) => result,
                Err(err) => {
                    let _ = compressed_tx.send(Err(err));
                    break;
                }
            };
            let processed = ProcessedChunk {
                index: chunk.index,
                offset: chunk.offset,
                length: chunk.length,
                hash,
                compressed,
            };
            if compressed_tx.send(Ok(processed)).is_err() {
                break;
            }
        }
    });

    // Collector to preserve order
    let (collector_tx, collector_rx) = bounded::<Result<Vec<ProcessedChunk>, ChunkingError>>(1);
    let collector_handle = thread::spawn(move || {
        let mut expected_index = 0usize;
        let mut buffer = BTreeMap::new();
        let mut results = Vec::new();

        for message in compressed_rx {
            match message {
                Ok(chunk) => {
                    let _ = buffer.insert(chunk.index, chunk);
                    while let Some(next) = buffer.remove(&expected_index) {
                        results.push(next);
                        expected_index += 1;
                    }
                }
                Err(err) => {
                    let _ = collector_tx.send(Err(ChunkingError::Compression(err)));
                    return;
                }
            }
        }

        let _ = collector_tx.send(Ok(results));
    });

    // Producer executes on current thread to ensure back-pressure works with bounded channels
    stream_chunks(reader, min_size, avg_size, max_size, |chunk| {
        chunk_tx
            .send(chunk)
            .map_err(|err| ChunkingError::Io(std::io::Error::new(std::io::ErrorKind::BrokenPipe, err)))
    })?;

    drop(chunk_tx);
    
    // Handle thread panics properly
    hashing_handle
        .join()
        .map_err(|e| ChunkingError::ThreadPanic(format!("hashing thread panicked: {e:?}")))?;
    compression_handle
        .join()
        .map_err(|e| ChunkingError::ThreadPanic(format!("compression thread panicked: {e:?}")))?;

    let collected = collector_rx
        .recv()
        .map_err(|err| ChunkingError::Io(std::io::Error::new(std::io::ErrorKind::BrokenPipe, err)))?;

    collector_handle
        .join()
        .map_err(|e| ChunkingError::ThreadPanic(format!("collector thread panicked: {e:?}")))?;

    collected
}

fn stream_chunks<R: Read>(
    mut reader: R,
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
    mut on_chunk: impl FnMut(StreamChunk) -> Result<(), ChunkingError>,
) -> Result<(), ChunkingError> {
    #[allow(clippy::cast_possible_truncation)]
    let min = min_size.unwrap_or(16_384) as u32;
    #[allow(clippy::cast_possible_truncation)]
    let avg = avg_size.unwrap_or(65_536) as u32;
    #[allow(clippy::cast_possible_truncation)]
    let max = max_size.unwrap_or(262_144) as u32;

    let mut buffer = Vec::with_capacity(max as usize * 2);
    let mut eof = false;
    let mut base_offset = 0usize;
    let mut index = 0usize;

    while !eof || !buffer.is_empty() {
        if !eof {
            let mut temp = vec![0u8; max as usize];
            let read = reader.read(&mut temp)?;
            if read == 0 {
                eof = true;
            } else {
                buffer.extend_from_slice(&temp[..read]);
            }
        }

        if buffer.is_empty() {
            continue;
        }

        let chunker = FastCDC::new(&buffer, min, avg, max);
        let mut produced: Vec<_> = chunker.collect();
        let mut pending = if !eof {
            produced.pop()
        } else {
            None
        };

        for chunk in produced {
            validate_slice_bounds(buffer.len(), chunk.offset, chunk.length)?;
            let start = chunk.offset;
            let end = chunk.offset + chunk.length;
            let data = buffer[start..end].to_vec();
            on_chunk(StreamChunk {
                index,
                offset: base_offset + start,
                length: chunk.length,
                data,
            })?;
            index += 1;
        }

        if eof {
            if let Some(chunk) = pending.take() {
                validate_slice_bounds(buffer.len(), chunk.offset, chunk.length)?;
                let start = chunk.offset;
                let end = chunk.offset + chunk.length;
                let data = buffer[start..end].to_vec();
                on_chunk(StreamChunk {
                    index,
                    offset: base_offset + start,
                    length: chunk.length,
                    data,
                })?;
                index += 1;
                base_offset += end;
                buffer.clear();
                continue;
            }
        }

        let retain_start = pending.as_ref().map_or(buffer.len(), |chunk| chunk.offset);
        base_offset += retain_start;

        if retain_start > 0 {
            buffer = buffer.split_off(retain_start);
        }
    }

    Ok(())
}
