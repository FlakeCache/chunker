#![no_main]
use libfuzzer_sys::fuzz_target;
use chunker::fastcdc::FastCDC;

fuzz_target!(|data: &[u8]| {
    if data.len() < 1024 {
        return;
    }
    let chunks = FastCDC::new(data, 1024, 4096, 16384);
    let mut reconstructed = Vec::with_capacity(data.len());
    for chunk in chunks {
        reconstructed.extend_from_slice(&data[chunk.offset..chunk.offset + chunk.length]);
    }
    assert_eq!(data, reconstructed.as_slice());
});
