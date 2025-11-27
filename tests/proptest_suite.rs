use chunker::compression::{compress_zstd, decompress_zstd};
use chunker::signing::{generate_keypair, sign_data, verify_signature};
use fastcdc::v2020::FastCDC;
use proptest::prelude::*;

fn zstd_roundtrip_ok(data: &[u8]) -> bool {
    compress_zstd(data, Some(3)).is_ok_and(|compressed| {
        decompress_zstd(&compressed).is_ok_and(|decompressed| data == decompressed)
    })
}

fn signing_roundtrip_ok(data: &[u8]) -> bool {
    let (secret, public) = generate_keypair();
    sign_data(data, &secret).is_ok_and(|sig| verify_signature(data, &sig, &public).is_ok())
}

proptest! {
    #[test]
    fn test_zstd_roundtrip(data in proptest::collection::vec(any::<u8>(), 0..10 * 1024)) {
        prop_assert!(zstd_roundtrip_ok(&data));
    }

    #[test]
    fn test_fastcdc_chunking_invariants(data in proptest::collection::vec(any::<u8>(), 1024..10 * 1024)) {
        let chunks = FastCDC::new(&data, 1024, 4096, 16384);
        let mut reconstructed = Vec::new();
        for chunk in chunks {
            reconstructed.extend_from_slice(&data[chunk.offset..chunk.offset + chunk.length]);
        }
        prop_assert_eq!(data, reconstructed);
    }

    #[test]
    fn test_signing_verification(data in proptest::collection::vec(any::<u8>(), 0..1024)) {
        prop_assert!(signing_roundtrip_ok(&data));
    }
}
