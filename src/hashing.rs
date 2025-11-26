use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "nif", derive(rustler::NifUnitEnum))]
pub enum HashAlgorithm {
    Sha256,
    #[cfg(feature = "blake3")]
    Blake3,
}

impl Default for HashAlgorithm {
    fn default() -> Self {
        Self::Sha256
    }
}

#[derive(Debug, thiserror::Error, Clone, Copy)]
pub enum HashingError {
    #[error("invalid_base32_character")]
    InvalidCharacter,
}

// Nix uses a custom base32 alphabet
pub const NIX_BASE32_ALPHABET: &[u8] = b"0123456789abcdfghijklmnpqrsvwxyz";

/// Compute SHA256 hash of data
/// Returns: hex string
pub fn sha256_hash(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    hex::encode(result)
}

/// Compute BLAKE3 hash of data
/// Returns: hex string
#[cfg(feature = "blake3")]
pub fn blake3_hash(data: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(data);
    let result = hasher.finalize();
    result.to_hex().to_string()
}

/// Compute hash of data using the specified algorithm, defaulting to SHA-256.
pub fn hash_bytes(data: &[u8], algorithm: HashAlgorithm) -> String {
    match algorithm {
        HashAlgorithm::Sha256 => sha256_hash(data),
        #[cfg(feature = "blake3")]
        HashAlgorithm::Blake3 => blake3_hash(data),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_bytes_defaults() {
        let sha_result = sha256_hash(b"hash me");
        assert_eq!(hash_bytes(b"hash me", HashAlgorithm::Sha256), sha_result);
    }

    #[cfg(feature = "blake3")]
    #[test]
    fn hash_bytes_blake3() {
        let blake3_result = blake3_hash(b"hash me");
        assert_eq!(hash_bytes(b"hash me", HashAlgorithm::Blake3), blake3_result);
    }
}

/// Encode data to Nix's base32 format
/// Nix uses a custom alphabet for base32 encoding
pub fn nix_base32_encode(data: &[u8]) -> String {
    let bytes = data;
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
pub fn nix_base32_decode(encoded: &str) -> Result<Vec<u8>, HashingError> {
    let mut result = Vec::new();
    let mut bits = 0u64;
    let mut bit_count = 0u8;

    for c in encoded.chars() {
        let value = NIX_BASE32_ALPHABET
            .iter()
            .position(|&b| b == c as u8)
            .ok_or(HashingError::InvalidCharacter)? as u64;

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
