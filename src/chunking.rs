use std::ops::Range;

use fastcdc::v2020::FastCDC;
use sha2::{Digest, Sha256};

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

/// Gear table constants generated from a simple XorShift64 routine to produce
/// deterministic pseudo-random values for coarse scanning.
const fn generate_gear_table() -> [u64; 256] {
    let mut table = [0u64; 256];
    let mut i = 0;
    let mut state = 0x243f_6a88_85a3_08d3u64;
    while i < 256 {
        state ^= state << 7;
        state ^= state >> 9;
        state ^= state << 8;
        table[i] = state;
        i += 1;
    }
    table
}

const GEAR_TABLE: [u64; 256] = generate_gear_table();

/// Controls how chunk boundaries are computed.
#[derive(Debug, Clone, Copy)]
pub enum ChunkingStrategy {
    /// Run FastCDC once across the entire input.
    FastCdc {
        min_size: Option<usize>,
        avg_size: Option<usize>,
        max_size: Option<usize>,
    },
    /// Perform a coarse gear-mask scan to isolate candidate regions, then run
    /// FastCDC only within those regions using the provided fine-grained
    /// configuration. Regions outside the coarse matches are emitted as single
    /// coarse chunks to keep boundaries deterministic.
    TwoTier {
        coarse_window: usize,
        coarse_mask: u64,
        fine_min: usize,
        fine_avg: usize,
        fine_max: usize,
    },
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
    chunk_data_with_strategy(
        data,
        ChunkingStrategy::FastCdc {
            min_size,
            avg_size,
            max_size,
        },
    )
}

/// Main entry point for chunking with configurable strategy.
pub fn chunk_data_with_strategy(
    data: &[u8],
    strategy: ChunkingStrategy,
) -> Result<Vec<(String, usize, usize)>, ChunkingError> {
    match strategy {
        ChunkingStrategy::FastCdc {
            min_size,
            avg_size,
            max_size,
        } => run_fastcdc(data, min_size, avg_size, max_size, 0),
        ChunkingStrategy::TwoTier {
            coarse_window,
            coarse_mask,
            fine_min,
            fine_avg,
            fine_max,
        } => two_tier_chunk(
            data,
            coarse_window,
            coarse_mask,
            fine_min,
            fine_avg,
            fine_max,
        ),
    }
}

fn run_fastcdc(
    data: &[u8],
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
    offset_base: usize,
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

        let global_offset = offset_base + chunk.offset;

        // Compute SHA256 hash of chunk
        let mut hasher = Sha256::new();
        hasher.update(&data[chunk.offset..chunk.offset + chunk.length]);
        let hash = hasher.finalize();
        let hash_hex = hex::encode(hash);

        chunks.push((hash_hex, global_offset, chunk.length));
    }

    Ok(chunks)
}

fn coarse_mask_ranges(data: &[u8], window: usize, mask: u64) -> Vec<Range<usize>> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut ranges: Vec<Range<usize>> = Vec::new();
    let mut hash = 0u64;
    for (idx, byte) in data.iter().enumerate() {
        hash = hash
            .wrapping_shl(1)
            .wrapping_add(GEAR_TABLE[*byte as usize]);

        if idx + 1 >= window && (hash & mask) == 0 {
            let start = idx + 1 - window;
            let end = idx + 1;
            ranges.push(start..end);
            hash = 0; // reset to look for the next coarse region
        }
    }

    merge_ranges(ranges)
}

fn merge_ranges(mut ranges: Vec<Range<usize>>) -> Vec<Range<usize>> {
    if ranges.is_empty() {
        return ranges;
    }

    ranges.sort_by_key(|range| range.start);
    let mut merged = Vec::with_capacity(ranges.len());
    let mut current = ranges[0].clone();

    for range in ranges.into_iter().skip(1) {
        if range.start <= current.end {
            current.end = current.end.max(range.end);
        } else {
            merged.push(current);
            current = range;
        }
    }

    merged.push(current);
    merged
}

fn two_tier_chunk(
    data: &[u8],
    coarse_window: usize,
    coarse_mask: u64,
    fine_min: usize,
    fine_avg: usize,
    fine_max: usize,
) -> Result<Vec<(String, usize, usize)>, ChunkingError> {
    if coarse_window == 0 {
        return run_fastcdc(data, Some(fine_min), Some(fine_avg), Some(fine_max), 0);
    }

    let coarse_matches = coarse_mask_ranges(data, coarse_window, coarse_mask);
    if coarse_matches.is_empty() {
        return run_fastcdc(data, Some(fine_min), Some(fine_avg), Some(fine_max), 0);
    }

    let mut chunks = Vec::new();
    let mut cursor = 0usize;

    for range in coarse_matches {
        // Emit coarse chunk for unchanged region before the flagged area
        if cursor < range.start {
            validate_slice_bounds(data.len(), cursor, range.start - cursor)?;
            let mut hasher = Sha256::new();
            hasher.update(&data[cursor..range.start]);
            let hash = hasher.finalize();
            chunks.push((hex::encode(hash), cursor, range.start - cursor));
        }

        let expanded_start = range.start.saturating_sub(coarse_window);
        let expanded_end = (range.end + coarse_window).min(data.len());
        validate_slice_bounds(data.len(), expanded_start, expanded_end - expanded_start)?;
        let expanded_slice = &data[expanded_start..expanded_end];
        let fine_chunks = run_fastcdc(
            expanded_slice,
            Some(fine_min),
            Some(fine_avg),
            Some(fine_max),
            expanded_start,
        )?;

        // Keep only the chunks whose boundaries land in the original coarse match
        let mut covered = range.start;
        for (hash, offset, length) in fine_chunks {
            if offset >= range.start && offset + length <= range.end {
                if offset > covered {
                    validate_slice_bounds(data.len(), covered, offset - covered)?;
                    let mut hasher = Sha256::new();
                    hasher.update(&data[covered..offset]);
                    let hash = hasher.finalize();
                    chunks.push((hex::encode(hash), covered, offset - covered));
                }
                chunks.push((hash, offset, length));
                covered = offset + length;
            }
        }

        if covered < range.end {
            validate_slice_bounds(data.len(), covered, range.end - covered)?;
            let mut hasher = Sha256::new();
            hasher.update(&data[covered..range.end]);
            let hash = hasher.finalize();
            chunks.push((hex::encode(hash), covered, range.end - covered));
        }

        cursor = range.end;
    }

    if cursor < data.len() {
        validate_slice_bounds(data.len(), cursor, data.len() - cursor)?;
        let mut hasher = Sha256::new();
        hasher.update(&data[cursor..]);
        let hash = hasher.finalize();
        chunks.push((hex::encode(hash), cursor, data.len() - cursor));
    }

    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn coarse_scanner_marks_regions() {
        let data: Vec<u8> = (0..8_192).map(|i| (i % 251) as u8).collect();
        let ranges = coarse_mask_ranges(&data, 256, 0);
        assert!(!ranges.is_empty());
        assert!(ranges.iter().all(|r| r.end > r.start));
    }

    #[test]
    fn two_tier_preserves_fine_boundaries_in_changed_regions() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut data = Vec::new();
        data.extend(std::iter::repeat(0u8).take(4_096));
        data.extend((0..8_192).map(|_| rng.gen::<u8>()));
        data.extend(std::iter::repeat(0u8).take(2_048));
        data.extend((0..4_096).map(|_| rng.gen::<u8>()));
        data.extend(std::iter::repeat(0u8).take(1_024));

        let coarse_window = 4_096;
        let coarse_mask = 0x1fff;
        let fine_min = 1_024;
        let fine_avg = 2_048;
        let fine_max = 4_096;

        let coarse_ranges = coarse_mask_ranges(&data, coarse_window, coarse_mask);
        assert!(!coarse_ranges.is_empty());

        let single_pass = chunk_data_with_strategy(
            &data,
            ChunkingStrategy::FastCdc {
                min_size: Some(fine_min),
                avg_size: Some(fine_avg),
                max_size: Some(fine_max),
            },
        )
        .expect("single-pass chunking should succeed");

        let two_tier = chunk_data_with_strategy(
            &data,
            ChunkingStrategy::TwoTier {
                coarse_window,
                coarse_mask,
                fine_min,
                fine_avg,
                fine_max,
            },
        )
        .expect("two-tier chunking should succeed");

        fn boundaries_in_ranges(
            chunks: &[(String, usize, usize)],
            ranges: &[Range<usize>],
        ) -> Vec<(usize, usize)> {
            let mut result = Vec::new();
            for (.., offset, length) in chunks {
                let end = *offset + *length;
                if ranges
                    .iter()
                    .any(|range| range.start <= *offset && end <= range.end)
                {
                    result.push((*offset, *length));
                }
            }
            result
        }

        let single_flagged = boundaries_in_ranges(&single_pass, &coarse_ranges);
        let two_tier_flagged = boundaries_in_ranges(&two_tier, &coarse_ranges);

        assert!(!single_flagged.is_empty());
        assert!(!two_tier_flagged.is_empty());
        let overlap = two_tier_flagged
            .iter()
            .filter(|boundary| single_flagged.contains(boundary))
            .count();
        assert!(overlap > 0);

        let second_pass = chunk_data_with_strategy(
            &data,
            ChunkingStrategy::TwoTier {
                coarse_window,
                coarse_mask,
                fine_min,
                fine_avg,
                fine_max,
            },
        )
        .expect("second two-tier chunking should succeed");
        assert_eq!(two_tier, second_pass);

        let total_single: usize = single_pass.iter().map(|(_, _, len)| *len).sum();
        let total_two_tier: usize = two_tier.iter().map(|(_, _, len)| *len).sum();
        assert_eq!(total_single, data.len());
        assert_eq!(total_two_tier, data.len());
    }
}
