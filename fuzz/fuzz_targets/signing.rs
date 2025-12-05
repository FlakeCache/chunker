#![no_main]
use libfuzzer_sys::fuzz_target;
use chunker::signing::{generate_keypair, sign_data, verify_signature};

fuzz_target!(|data: &[u8]| {
    // Generate a keypair (should not panic)
    let (secret_b64, public_b64) = generate_keypair();

    // Attempt to sign arbitrary data
    if let Ok(signature_b64) = sign_data(data, &secret_b64) {
        // Verify the signature - should always succeed if we just signed it
        let _ = verify_signature(data, &signature_b64, &public_b64);
    }

    // Also test verification with arbitrary signature data
    // This ensures we don't panic on malformed signatures or data
    let _ = verify_signature(data, "invalidbase64!@#$", &public_b64);

    // Test with malformed public key
    let _ = sign_data(data, "notavalidbase64key");
    let _ = verify_signature(data, &"fakesignature".repeat(10), "notavalidbase64key");
});
