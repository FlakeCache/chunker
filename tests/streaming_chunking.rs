use chunker::chunking;
use rand::{rngs::StdRng, RngCore, SeedableRng};
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

    let streaming = chunking::chunk_stream(reader, None, None, None)?;
    let in_memory = chunking::chunk_data(&data, None, None, None)?;

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

    let streaming = chunking::chunk_stream(reader, min, avg, max)?;
    let in_memory = chunking::chunk_data(&data, min, avg, max)?;

    assert_eq!(in_memory, streaming);
    Ok(())
}
