use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("serialization_failed: {0}")]
    Serialization(String),
    #[error("deserialization_failed: {0}")]
    Deserialization(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HashAlgorithm {
    #[serde(rename = "sha256")]
    Sha256,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionCodec {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "zstd")]
    Zstd,
    #[serde(rename = "xz")]
    Xz,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompressionSettings {
    pub codec: CompressionCodec,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub level: Option<i32>,
}

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            codec: CompressionCodec::None,
            level: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CdcStrategy {
    #[serde(rename = "fastcdc")]
    FastCdc {
        min: u32,
        avg: u32,
        max: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkDescriptor {
    pub offset: u64,
    pub length: u64,
    pub hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u8,
    pub hash_algorithm: HashAlgorithm,
    pub cdc_strategy: CdcStrategy,
    pub compression: CompressionSettings,
    pub merkle_root: String,
    pub chunks: Vec<ChunkDescriptor>,
}

impl Manifest {
    pub fn to_canonical_json_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        serde_json::to_vec(self).map_err(|err| ManifestError::Serialization(err.to_string()))
    }

    pub fn to_canonical_cbor_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        serde_cbor::to_vec(self).map_err(|err| ManifestError::Serialization(err.to_string()))
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, ManifestError> {
        serde_json::from_slice(bytes)
            .map_err(|err| ManifestError::Deserialization(err.to_string()))
    }

    pub fn from_cbor_bytes(bytes: &[u8]) -> Result<Self, ManifestError> {
        serde_cbor::from_slice(bytes)
            .map_err(|err| ManifestError::Deserialization(err.to_string()))
    }
}

pub fn compute_merkle_root(hashes: &[Vec<u8>]) -> Option<Vec<u8>> {
    if hashes.is_empty() {
        return None;
    }

    let mut level: Vec<Vec<u8>> = hashes.to_vec();

    while level.len() > 1 {
        let mut next_level: Vec<Vec<u8>> = Vec::new();

        let mut idx = 0;
        while idx < level.len() {
            let left = &level[idx];
            let right = if idx + 1 < level.len() {
                &level[idx + 1]
            } else {
                left
            };

            let mut hasher = Sha256::new();
            hasher.update(left);
            hasher.update(right);
            next_level.push(hasher.finalize().to_vec());

            idx += 2;
        }

        level = next_level;
    }

    level.pop()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merkle_root_matches_manual_pairing() {
        let leaves = vec![
            vec![0u8; 32],
            vec![1u8; 32],
            vec![2u8; 32],
            vec![3u8; 32],
        ];

        let root = compute_merkle_root(&leaves).expect("merkle root");

        let mut level_one: Vec<Vec<u8>> = Vec::new();
        for pair in leaves.chunks(2) {
            let mut hasher = Sha256::new();
            hasher.update(&pair[0]);
            hasher.update(&pair[1]);
            level_one.push(hasher.finalize().to_vec());
        }

        let mut hasher = Sha256::new();
        hasher.update(&level_one[0]);
        hasher.update(&level_one[1]);
        let expected = hasher.finalize().to_vec();

        assert_eq!(root, expected);
    }

    #[test]
    fn json_round_trip_is_deterministic() {
        let manifest = Manifest {
            version: 1,
            hash_algorithm: HashAlgorithm::Sha256,
            cdc_strategy: CdcStrategy::FastCdc {
                min: 1,
                avg: 2,
                max: 3,
            },
            compression: CompressionSettings {
                codec: CompressionCodec::Zstd,
                level: Some(3),
            },
            merkle_root: "abcd".into(),
            chunks: vec![ChunkDescriptor {
                offset: 0,
                length: 4,
                hash: "deadbeef".into(),
            }],
        };

        let first = manifest
            .to_canonical_json_bytes()
            .expect("json serialization");
        let decoded = Manifest::from_json_bytes(&first).expect("json parse");
        let second = decoded
            .to_canonical_json_bytes()
            .expect("json serialization");

        assert_eq!(first, second);
    }

    #[test]
    fn cbor_round_trip_is_deterministic() {
        let manifest = Manifest {
            version: 1,
            hash_algorithm: HashAlgorithm::Sha256,
            cdc_strategy: CdcStrategy::FastCdc {
                min: 1,
                avg: 2,
                max: 3,
            },
            compression: CompressionSettings::default(),
            merkle_root: "abcd".into(),
            chunks: vec![ChunkDescriptor {
                offset: 0,
                length: 4,
                hash: "deadbeef".into(),
            }],
        };

        let first = manifest
            .to_canonical_cbor_bytes()
            .expect("cbor serialization");
        let decoded = Manifest::from_cbor_bytes(&first).expect("cbor parse");
        let second = decoded
            .to_canonical_cbor_bytes()
            .expect("cbor serialization");

        assert_eq!(first, second);
    }
}
