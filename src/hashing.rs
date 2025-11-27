use sha2::{Digest, Sha256};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};

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
pub fn blake3_hash(data: &[u8]) -> String {
    let hash = blake3::hash(data);
    hash.to_hex().to_string()
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

#[derive(Debug, Clone)]
pub struct HashJob {
    pub index: usize,
    pub offset: usize,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct HashResult {
    pub index: usize,
    pub offset: usize,
    pub length: usize,
    pub digest: String,
}

/// Spawn a bounded hashing worker that preserves submission order.
/// The worker terminates when it receives `None` via the job channel.
/// Returns (sender, receiver, worker_handle) for panic detection and synchronization.
pub fn spawn_hashing_worker(
    bound: usize,
) -> (
    SyncSender<Option<HashJob>>,
    Receiver<HashResult>,
    std::thread::JoinHandle<()>,
) {
    let (job_tx, job_rx) = sync_channel(bound);
    let (result_tx, result_rx) = sync_channel(bound);

    let handle = std::thread::spawn(move || {
        while let Ok(message) = job_rx.recv() {
            let Some(job): Option<HashJob> = message else {
                break;
            };
            let digest = sha256_hash(&job.data);
            let result = HashResult {
                index: job.index,
                offset: job.offset,
                length: job.data.len(),
                digest,
            };
            if result_tx.send(result).is_err() {
                break;
            }
        }
    });

    (job_tx, result_rx, handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nix_base32_encode_known_values() {
        // Known test vectors from Nix source or documentation
        assert_eq!(nix_base32_encode(b""), "");
        assert_eq!(nix_base32_encode(b"foo"), "0z11");
    }

    #[test]
    fn test_blake3_basic() {
        let data = b"hello world";
        let hash = blake3_hash(data);
        // Known BLAKE3 hash for "hello world"
        assert_eq!(
            hash,
            "d74981efa70a0c880b8d8c1985d075dbcbf679b99a5f9914e5aaf96b831a9e24"
        );
    }
}

#[cfg(test)]
mod extra_tests {
    use super::*;

    #[test]
    fn test_nix_base32_roundtrip() -> Result<(), HashingError> {
        let data = b"hello world";
        let encoded = nix_base32_encode(data);
        let decoded = nix_base32_decode(&encoded)?;
        assert_eq!(data, decoded.as_slice());
        Ok(())
    }

    #[test]
    fn test_nix_base32_decode_errors() {
        assert!(matches!(nix_base32_decode("invalid!"), Err(HashingError::InvalidCharacter)));
    }

    #[test]
    fn test_hashing_worker() -> Result<(), String> {
        let (tx, rx, handle) = spawn_hashing_worker(10);
        
        let data = b"worker test data";
        tx.send(Some(HashJob {
            index: 0,
            offset: 0,
            data: data.to_vec(),
        }))
        .map_err(|err| err.to_string())?;
        tx.send(None).map_err(|err| err.to_string())?;

        let result = rx.recv().map_err(|err| err.to_string())?;
        assert_eq!(result.index, 0);
        assert_eq!(result.digest, sha256_hash(data));

        handle.join().map_err(|_| "worker panicked".to_string())?;
        Ok(())
    }
}
