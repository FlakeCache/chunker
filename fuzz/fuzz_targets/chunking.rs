#![no_main]
use libfuzzer_sys::fuzz_target;
use chunker::chunking::ChunkStream;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    // Use a Cursor to simulate a stream reader
    let reader = Cursor::new(data);
    
    // Initialize ChunkStream with standard options
    // This exercises the internal BytesMut slab buffer logic
    let stream = ChunkStream::new(reader, None, None, None).expect("Failed to create stream");
    
    let mut reconstructed = Vec::with_capacity(data.len());
    let mut total_len = 0;

    for chunk_res in stream {
        let chunk = chunk_res.expect("Chunking failed");
        
        // Verify offset continuity
        assert_eq!(chunk.offset, total_len as u64, "Offset mismatch");
        
        // Verify payload matches source data
        let start = chunk.offset as usize;
        let end = start + chunk.length;
        assert_eq!(&data[start..end], chunk.payload.as_ref(), "Payload mismatch");
        
        reconstructed.extend_from_slice(&chunk.payload);
        total_len += chunk.length;
    }

    // Verify total reconstruction
    assert_eq!(data, reconstructed.as_slice(), "Reconstruction mismatch");
    assert_eq!(total_len, data.len(), "Length mismatch");
});
