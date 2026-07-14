use chunker::chunking::{self, HashAlgorithm};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::fs::File;
use std::io::{BufReader, Write};

#[test]
fn streaming_matches_in_memory_for_large_fixture() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut data = vec![0u8; 2_500_000];
    rng.fill_bytes(&mut data);

    let mut temp = tempfile::NamedTempFile::new()?;
    temp.write_all(&data)?;
    temp.flush()?;

    let file = File::open(temp.path())?;
    let reader = BufReader::new(file);

    let streaming =
        chunking::chunk_stream_with_hash(reader, None, None, None, HashAlgorithm::Blake3)?;
    let in_memory = chunking::chunk_data_with_hash(&data, None, None, None, HashAlgorithm::Blake3)?;

    assert_eq!(in_memory, streaming);
    Ok(())
}

#[test]
fn streaming_respects_custom_boundaries() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(84);
    let mut data = vec![0u8; 3_000_000];
    rng.fill_bytes(&mut data);

    let mut temp = tempfile::NamedTempFile::new()?;
    temp.write_all(&data)?;
    temp.flush()?;

    let file = File::open(temp.path())?;
    let reader = BufReader::new(file);

    let min = Some(8_192usize);
    let avg = Some(32_768usize);
    let max = Some(131_072usize);

    let streaming = chunking::chunk_stream_with_hash(reader, min, avg, max, HashAlgorithm::Blake3)?;
    let in_memory = chunking::chunk_data_with_hash(&data, min, avg, max, HashAlgorithm::Blake3)?;

    assert_eq!(in_memory, streaming);
    Ok(())
}

// Push-fed (incremental) chunking: the source is delivered in arbitrary slice
// sizes (as an upload streams in over the socket) rather than as one buffer.
// Contract: for ANY slice boundary, the emitted chunks must be byte-identical
// to chunking the whole artifact in memory, and the emitted payloads must
// reassemble to the exact original bytes (so each chunk can be stored inline).
#[test]
fn push_matches_in_memory_across_slice_sizes() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(1234);
    let mut data = vec![0u8; 1_500_000];
    rng.fill_bytes(&mut data);

    // Small chunk params keep the retained buffer tiny (<=max_size), so even
    // small feed slices stay cheap while still exercising ~40+ boundaries.
    let (min, avg, max) = (Some(8_192usize), Some(32_768usize), Some(131_072usize));
    let expected = chunking::chunk_data_with_hash(&data, min, avg, max, HashAlgorithm::Blake3)?;

    // 512B..whole-in-one-push (>data) covers backpressure-tiny feeds through
    // single-shot uploads. A single push emits multiple chunks when it spans
    // several boundaries.
    for slice in [512usize, 8_192, 65_536, 400_000, 1_500_000] {
        let mut pc = chunking::PushChunker::new(min, avg, max, HashAlgorithm::Blake3)?;
        let mut got = Vec::new();
        let mut reassembled: Vec<u8> = Vec::new();

        for part in data.chunks(slice) {
            for chunk in pc.push(part)? {
                reassembled.extend_from_slice(&chunk.payload);
                got.push(chunk);
            }
        }
        for chunk in pc.finish()? {
            reassembled.extend_from_slice(&chunk.payload);
            got.push(chunk);
        }

        assert_eq!(got, expected, "chunk boundaries diverged for slice={slice}");
        assert_eq!(
            reassembled, data,
            "payload reassembly diverged for slice={slice}"
        );

        // Per-chunk integrity (ChunkMetadata::PartialEq ignores payloads, so the
        // equivalence assert above does NOT check them): each payload's length
        // matches, equals the source bytes at the chunk's absolute offset, and no
        // chunk exceeds max_size.
        for chunk in &got {
            let off = usize::try_from(chunk.offset)?;
            assert_eq!(chunk.payload.len(), chunk.length, "payload len != length");
            assert!(
                chunk.length <= 131_072,
                "chunk {} exceeds max_size",
                chunk.length
            );
            assert_eq!(
                chunk.payload.as_ref(),
                &data[off..off + chunk.length],
                "payload != source slice at offset {off}"
            );
        }
    }
    Ok(())
}

// Empty input must finish cleanly with zero chunks (no phantom trailing chunk).
#[test]
fn push_empty_input_yields_no_chunks() -> Result<(), Box<dyn std::error::Error>> {
    let mut pc = chunking::PushChunker::new(None, None, None, HashAlgorithm::Sha256)?;
    assert!(pc.push(b"")?.is_empty());
    assert!(pc.finish()?.is_empty());
    Ok(())
}

// Nonuniform feed schedule (1-byte through > max_size steps) with an empty push
// interleaved before every real push — the empties must be no-ops and the result
// must still equal in-memory chunking.
#[test]
fn push_matches_in_memory_nonuniform_with_empties() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(99);
    let mut data = vec![0u8; 800_000];
    rng.fill_bytes(&mut data);

    let (min, avg, max) = (Some(8_192usize), Some(32_768usize), Some(131_072usize));
    let expected = chunking::chunk_data_with_hash(&data, min, avg, max, HashAlgorithm::Sha256)?;

    let mut pc = chunking::PushChunker::new(min, avg, max, HashAlgorithm::Sha256)?;
    let mut got = Vec::new();
    let steps = [1usize, 7, 33, 1, 60_000, 3, 131_073, 500, 250_000, 9];
    let (mut pos, mut i) = (0usize, 0usize);
    while pos < data.len() {
        assert!(pc.push(b"")?.is_empty(), "empty push must yield nothing");
        let step = steps[i % steps.len()].min(data.len() - pos);
        i += 1;
        got.extend(pc.push(&data[pos..pos + step])?);
        pos += step;
    }
    got.extend(pc.finish()?);

    assert_eq!(got, expected, "nonuniform feed with empties diverged");
    Ok(())
}

// After finish(), further push() must error (not panic, not silently accept).
#[test]
fn push_after_finish_errors() -> Result<(), Box<dyn std::error::Error>> {
    let mut pc = chunking::PushChunker::new(None, None, None, HashAlgorithm::Sha256)?;
    let _ = pc.push(b"hello streaming world")?;
    let _ = pc.finish()?;
    assert!(pc.push(b"more").is_err(), "push after finish must error");
    Ok(())
}

// Options outside FastCDC v2020's supported ranges must be rejected at new()
// (previously they were accepted and later panicked inside FastCDC, poisoning the
// NIF resource mutex).
#[test]
fn push_chunker_rejects_out_of_range_options() {
    // avg 64 is below FastCDC's AVERAGE_MIN (256).
    assert!(
        chunking::PushChunker::new(Some(64), Some(64), Some(64), HashAlgorithm::Sha256).is_err(),
        "sub-minimum avg_size must be rejected"
    );
    // max 64 MiB is above FastCDC's MAXIMUM_MAX (16 MiB).
    assert!(
        chunking::PushChunker::new(
            Some(64),
            Some(1024),
            Some(64 * 1024 * 1024),
            HashAlgorithm::Sha256
        )
        .is_err(),
        "over-maximum max_size must be rejected"
    );
}
