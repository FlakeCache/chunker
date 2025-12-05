#![no_main]
use libfuzzer_sys::fuzz_target;
use chunker::decompress_auto;

fuzz_target!(|data: &[u8]| {
    // Fuzz the auto-decompression logic with arbitrary binary input
    // This ensures panic safety when handling malformed compressed data
    let _ = decompress_auto(data);
});
