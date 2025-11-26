use std::collections::{HashMap, HashSet};

use crate::chunking::{chunk_data, ChunkingError};
use crate::compression::{compress_zstd, CompressionError};

#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    #[error(transparent)]
    Chunking(#[from] ChunkingError),
    #[error(transparent)]
    Compression(#[from] CompressionError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowConfig {
    pub min: usize,
    pub avg: usize,
    pub max: usize,
}

impl WindowConfig {
    pub fn with_defaults() -> Self {
        Self {
            min: 16_384,
            avg: 65_536,
            max: 262_144,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Artifact<'a> {
    pub path: &'a str,
    pub class: &'a str,
    pub data: &'a [u8],
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MetricSnapshot {
    pub reused_chunks: usize,
    pub total_chunks: usize,
    pub compressed_bytes: usize,
    pub uncompressed_bytes: usize,
}

impl MetricSnapshot {
    pub fn reuse_ratio(&self) -> f64 {
        if self.total_chunks == 0 {
            return 0.0;
        }
        self.reused_chunks as f64 / self.total_chunks as f64
    }

    pub fn compression_ratio(&self) -> f64 {
        if self.uncompressed_bytes == 0 {
            return 0.0;
        }
        self.compressed_bytes as f64 / self.uncompressed_bytes as f64
    }

    pub fn merge(&mut self, other: &MetricSnapshot) {
        self.reused_chunks += other.reused_chunks;
        self.total_chunks += other.total_chunks;
        self.compressed_bytes += other.compressed_bytes;
        self.uncompressed_bytes += other.uncompressed_bytes;
    }
}

#[derive(Clone, Debug, Default)]
pub struct IngestMetrics {
    pub per_path: HashMap<String, MetricSnapshot>,
    pub per_class: HashMap<String, MetricSnapshot>,
}

impl IngestMetrics {
    fn record(&mut self, path: &str, class: &str, snapshot: MetricSnapshot) {
        let _ = self.per_path.insert(path.to_string(), snapshot);

        let entry = self
            .per_class
            .entry(class.to_string())
            .or_insert_with(MetricSnapshot::default);
        entry.merge(&snapshot);
    }

    pub fn class_metrics(&self, class: &str) -> Option<&MetricSnapshot> {
        self.per_class.get(class)
    }
}

#[derive(Debug)]
pub struct Ingestor {
    window_config: WindowConfig,
    chunk_index: HashSet<String>,
    pub metrics: IngestMetrics,
}

impl Ingestor {
    pub fn new(window_config: WindowConfig) -> Self {
        Self {
            window_config,
            chunk_index: HashSet::new(),
            metrics: IngestMetrics::default(),
        }
    }

    pub fn ingest(&mut self, artifact: Artifact<'_>) -> Result<MetricSnapshot, IngestError> {
        let cfg = self.window_config;
        let chunks = chunk_data(artifact.data, Some(cfg.min), Some(cfg.avg), Some(cfg.max))?;

        let reused_chunks = chunks
            .iter()
            .filter(|(hash, _, _)| self.chunk_index.contains(hash))
            .count();

        for (hash, _, _) in &chunks {
            let _ = self.chunk_index.insert(hash.clone());
        }

        let compressed = compress_zstd(artifact.data, Some(3))?;
        let snapshot = MetricSnapshot {
            reused_chunks,
            total_chunks: chunks.len(),
            compressed_bytes: compressed.len(),
            uncompressed_bytes: artifact.data.len(),
        };

        self.metrics
            .record(artifact.path, artifact.class, snapshot.clone());

        Ok(snapshot)
    }

    pub fn set_window_config(&mut self, window_config: WindowConfig) {
        self.window_config = window_config;
    }

    pub fn window_config(&self) -> WindowConfig {
        self.window_config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data(prefix: u8, len: usize) -> Vec<u8> {
        std::iter::repeat(prefix).take(len).collect()
    }

    #[test]
    fn records_metrics_per_path_and_class() {
        let mut ingestor = Ingestor::new(WindowConfig {
            min: 1024,
            avg: 2048,
            max: 4096,
        });

        let artifact_a = Artifact {
            path: "logs/app.log",
            class: "log",
            data: &sample_data(b'A', 10 * 1024),
        };

        let artifact_b = Artifact {
            path: "artifacts/bin",
            class: "binary",
            data: &sample_data(b'B', 8 * 1024),
        };

        let snapshot_a = ingestor.ingest(artifact_a).expect("ingest A");
        let snapshot_b = ingestor.ingest(artifact_b).expect("ingest B");

        assert!(snapshot_a.total_chunks > 0);
        assert!(snapshot_b.total_chunks > 0);

        let log_metrics = ingestor
            .metrics
            .per_path
            .get("logs/app.log")
            .expect("log metrics recorded");
        assert_eq!(log_metrics.total_chunks, snapshot_a.total_chunks);

        let class_metrics = ingestor
            .metrics
            .class_metrics("log")
            .expect("class metrics aggregated");
        assert_eq!(class_metrics.total_chunks, snapshot_a.total_chunks);
        assert_eq!(class_metrics.reused_chunks, snapshot_a.reused_chunks);
        assert_eq!(
            class_metrics.uncompressed_bytes,
            snapshot_a.uncompressed_bytes
        );
    }
}
