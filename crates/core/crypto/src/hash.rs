// SPDX-License-Identifier: MIT
// Derived from rvf-crypto in RuVector; see THIRD_PARTY_NOTICES.md.

//! SHAKE-256 hashing for content identifiers and witness binding.

use alloc::{vec, vec::Vec};
use shake::{ExtendableOutput, Shake256, Update, XofReader};

/// Compute a SHAKE-256 digest with an arbitrary output length.
#[must_use]
pub fn shake256_hash(data: &[u8], output_len: usize) -> Vec<u8> {
    let mut hasher = Shake256::default();
    hasher.update(data);
    let mut reader = hasher.finalize_xof();
    let mut output = vec![0_u8; output_len];
    reader.read(&mut output);
    output
}

/// Compute the first 128 bits of a SHAKE-256 digest.
#[must_use]
pub fn shake256_128(data: &[u8]) -> [u8; 16] {
    let mut hasher = Shake256::default();
    hasher.update(data);
    let mut reader = hasher.finalize_xof();
    let mut output = [0_u8; 16];
    reader.read(&mut output);
    output
}

/// Compute the first 256 bits of a SHAKE-256 digest.
#[must_use]
pub fn shake256_256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Shake256::default();
    hasher.update(data);
    let mut reader = hasher.finalize_xof();
    let mut output = [0_u8; 32];
    reader.read(&mut output);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_nist_empty_input_vector() {
        assert_eq!(
            shake256_256(b""),
            [
                0x46, 0xb9, 0xdd, 0x2b, 0x0b, 0xa8, 0x8d, 0x13, 0x23, 0x3b, 0x3f, 0xeb, 0x74, 0x3e,
                0xeb, 0x24, 0x3f, 0xcd, 0x52, 0xea, 0x62, 0xb8, 0x1b, 0x82, 0xb5, 0x0c, 0x27, 0x64,
                0x6e, 0xd5, 0x76, 0x2f,
            ]
        );
    }

    #[test]
    fn short_digest_is_long_digest_prefix() {
        assert_eq!(
            shake256_128(b"flakecache"),
            shake256_256(b"flakecache")[..16]
        );
    }
}
