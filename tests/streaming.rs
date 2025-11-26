use std::io::Cursor;

use chunker::chunking::{chunk_data, chunk_stream, process_stream_with_pipeline};
use chunker::compression::decompress_zstd;
use chunker::hashing::sha256_hash;

fn large_fixture() -> Vec<u8> {
    let mut data = Vec::with_capacity(2_000_000);
    for i in 0..2_000_000u32 {
        data.push((i % 251) as u8);
    }
    data
}

#[test]
fn streaming_matches_in_memory_boundaries_and_hashes() {
    let data = large_fixture();
    let min = 8_192usize;
    let avg = 32_768usize;
    let max = 131_072usize;

    let expected = chunk_data(&data, Some(min), Some(avg), Some(max)).unwrap();
    let streamed = chunk_stream(Cursor::new(data.clone()), Some(min), Some(avg), Some(max)).unwrap();

    assert_eq!(expected.len(), streamed.len());
    for (lhs, rhs) in expected.iter().zip(streamed.iter()) {
        assert_eq!(lhs, rhs);
    }
}

#[test]
fn pipeline_preserves_order_and_compression() {
    let data = large_fixture();
    let min = 16_384usize;
    let avg = 65_536usize;
    let max = 262_144usize;

    let processed = process_stream_with_pipeline(
        Cursor::new(data.clone()),
        Some(min),
        Some(avg),
        Some(max),
        Some(3),
        4,
    )
    .unwrap();

    for (expected_index, chunk) in processed.iter().enumerate() {
        assert_eq!(chunk.index, expected_index);
        let start = chunk.offset;
        let end = start + chunk.length;
        assert!(end <= data.len());

        let source_slice = &data[start..end];
        let decompressed = decompress_zstd(&chunk.compressed).unwrap();
        assert_eq!(decompressed, source_slice);

        let expected_hash = sha256_hash(source_slice);
        assert_eq!(expected_hash, chunk.hash);
    }
}
