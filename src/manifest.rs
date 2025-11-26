use serde::{Deserialize, Serialize};

use crate::compression::{CompressionMethod, CompressionSettings};

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("json_serialization_failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("cbor_serialization_failed: {0}")]
    Cbor(#[from] serde_cbor::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestFormat {
    Json,
    Cbor,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Manifest {
    pub chunks: Vec<ChunkRecord>,
    pub cdc: CdcParameters,
    pub compression: CompressionDescriptor,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChunkRecord {
    pub hash: String,
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CdcParameters {
    pub strategy: String,
    pub min_size: u64,
    pub avg_size: u64,
    pub max_size: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompressionDescriptor {
    pub method: CompressionMethod,
    pub level: Option<i32>,
}

impl From<CompressionSettings> for CompressionDescriptor {
    fn from(settings: CompressionSettings) -> Self {
        Self {
            method: settings.method,
            level: settings.level,
        }
    }
}

impl Manifest {
    pub fn to_bytes(&self, format: ManifestFormat) -> Result<Vec<u8>, ManifestError> {
        match format {
            ManifestFormat::Json => self.to_canonical_json_bytes(),
            ManifestFormat::Cbor => self.to_canonical_cbor_bytes(),
        }
    }

    fn to_canonical_json_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        let mut buffer = Vec::new();
        let mut serializer = serde_json::Serializer::new(&mut buffer);
        self.serialize(&mut serializer)?;
        Ok(buffer)
    }

    fn to_canonical_cbor_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        Ok(serde_cbor::to_vec(self)?)
    }
}
