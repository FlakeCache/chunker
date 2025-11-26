use rustler::{Binary, Error, NifResult};
use sha2::{Digest, Sha256};

// Nix uses a custom base32 alphabet
pub const NIX_BASE32_ALPHABET: &[u8] = b"0123456789abcdfghijklmnpqrsvwxyz";

/// Compute SHA256 hash of data
/// Returns: hex string
#[rustler::nif]
pub fn sha256_hash(data: Binary) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data.as_slice());
    let result = hasher.finalize();
    hex::encode(result)
}

/// Encode data to Nix's base32 format
/// Nix uses a custom alphabet for base32 encoding
#[rustler::nif]
pub fn nix_base32_encode(data: Binary) -> String {
    let bytes = data.as_slice();
    let mut result = String::new();

    // Nix base32 encoding
    let mut bits = 0u64;
    let mut bit_count = 0u8;

    for &byte in bytes {
        bits |= u64::from(byte) << bit_count;
        bit_count += 8;

        while bit_count >= 5 {
            let index = (bits & 0x1f) as usize;
            result.push(NIX_BASE32_ALPHABET[index] as char);
            bits >>= 5;
            bit_count -= 5;
        }
    }

    if bit_count > 0 {
        let index = (bits & 0x1f) as usize;
        result.push(NIX_BASE32_ALPHABET[index] as char);
    }

    result
}

/// Decode Nix base32 encoded string
#[rustler::nif]
pub fn nix_base32_decode(encoded: &str) -> NifResult<Vec<u8>> {
    let mut result = Vec::new();
    let mut bits = 0u64;
    let mut bit_count = 0u8;

    for c in encoded.chars() {
        let value = NIX_BASE32_ALPHABET
            .iter()
            .position(|&b| b == c as u8)
            .ok_or(Error::BadArg)? as u64;

        bits |= value << bit_count;
        bit_count += 5;

        while bit_count >= 8 {
            result.push((bits & 0xff) as u8);
            bits >>= 8;
            bit_count -= 8;
        }
    }

    Ok(result)
}
