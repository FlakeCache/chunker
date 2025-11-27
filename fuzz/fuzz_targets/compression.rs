#![no_main]
use libfuzzer_sys::fuzz_target;
use chunker::{compress_zstd, decompress_zstd};

fuzz_target!(|data: &[u8]| {
    if let Ok(compressed) = compress_zstd(data, 3) {
        if let Ok(decompressed) = decompress_zstd(&compressed) {
            assert_eq!(data, decompressed.as_slice());
        }
    }
});
