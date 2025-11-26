use sha2::{Digest, Sha256};

/// Supported hashing algorithms for chunking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgorithm {
    /// SHA-256 (default, canonical integrity hash)
    Sha256,
    /// BLAKE3 (faster, optional)
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
    let hash = blake3::hash(data);
    hash.to_hex().to_string()
}

/// Compute a hash using the selected algorithm
/// Defaults to SHA-256
pub fn hash_bytes(data: &[u8], algorithm: HashAlgorithm) -> String {
    match algorithm {
        HashAlgorithm::Sha256 => sha256_hash(data),
        #[cfg(feature = "blake3")]
        HashAlgorithm::Blake3 => blake3_hash(data),
    }
}

/// Parse a string hash algorithm identifier
///
/// Recognized values (case-insensitive):
/// - "sha256" (default)
/// - "blake3" (when the `blake3` feature is enabled)
pub fn parse_hash_algorithm(name: &str) -> Option<HashAlgorithm> {
    match name.to_ascii_lowercase().as_str() {
        "sha256" => Some(HashAlgorithm::Sha256),
        #[cfg(feature = "blake3")]
        "blake3" => Some(HashAlgorithm::Blake3),
        _ => None,
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
