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
pub fn spawn_hashing_worker(bound: usize) -> (SyncSender<Option<HashJob>>, Receiver<HashResult>) {
    let (job_tx, job_rx) = sync_channel(bound);
    let (result_tx, result_rx) = sync_channel(bound);

    let _ = std::thread::spawn(move || {
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

    (job_tx, result_rx)
}
