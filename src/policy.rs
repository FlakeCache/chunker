use std::collections::HashMap;

use crate::ingest::{MetricSnapshot, WindowConfig};

#[derive(Clone, Copy, Debug)]
pub struct WindowBounds {
    pub min: WindowConfig,
    pub max: WindowConfig,
}

impl WindowBounds {
    pub fn clamp(&self, config: WindowConfig) -> WindowConfig {
        WindowConfig {
            min: config.min.clamp(self.min.min, self.max.min),
            avg: config.avg.clamp(self.min.avg, self.max.avg),
            max: config.max.clamp(self.min.max, self.max.max),
        }
    }
}

#[derive(Debug)]
pub struct WindowPolicy {
    base: WindowConfig,
    bounds: WindowBounds,
    reuse_growth_threshold: f64,
    reuse_shrink_threshold: f64,
    compression_growth_threshold: f64,
    compression_shrink_threshold: f64,
    per_class: HashMap<String, WindowConfig>,
}

impl WindowPolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base: WindowConfig,
        bounds: WindowBounds,
        reuse_growth_threshold: f64,
        reuse_shrink_threshold: f64,
        compression_growth_threshold: f64,
        compression_shrink_threshold: f64,
    ) -> Self {
        Self {
            base,
            bounds,
            reuse_growth_threshold,
            reuse_shrink_threshold,
            compression_growth_threshold,
            compression_shrink_threshold,
            per_class: HashMap::new(),
        }
    }

    pub fn config_for_class(&self, class: &str) -> WindowConfig {
        *self.per_class.get(class).unwrap_or(&self.base)
    }

    pub fn observe(&mut self, class: &str, metrics: &MetricSnapshot) -> WindowConfig {
        let mut current = self.config_for_class(class);
        let reuse_ratio = metrics.reuse_ratio();
        let compression_ratio = metrics.compression_ratio();

        // Deterministic adjustments: only shrink/grow in fixed increments based on thresholds.
        if reuse_ratio < self.reuse_shrink_threshold
            || compression_ratio > self.compression_shrink_threshold
        {
            current = self.shrink(current);
        } else if reuse_ratio > self.reuse_growth_threshold
            && compression_ratio < self.compression_growth_threshold
        {
            current = self.grow(current);
        }

        current = self.bounds.clamp(current);
        let _ = self.per_class.insert(class.to_string(), current);
        current
    }

    fn shrink(&self, current: WindowConfig) -> WindowConfig {
        WindowConfig {
            min: (current.min / 2).max(self.bounds.min.min),
            avg: (current.avg * 3 / 4).max(self.bounds.min.avg),
            max: (current.max * 3 / 4).max(self.bounds.min.max),
        }
    }

    fn grow(&self, current: WindowConfig) -> WindowConfig {
        WindowConfig {
            min: (current.min * 3 / 2).min(self.bounds.max.min),
            avg: (current.avg * 3 / 2).min(self.bounds.max.avg),
            max: (current.max * 3 / 2).min(self.bounds.max.max),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::{Artifact, Ingestor};

    fn make_dataset(with_insertion: bool) -> Vec<u8> {
        let mut data: Vec<u8> = Vec::new();
        for i in 0..80 {
            let byte = if i % 9 == 0 { b'B' } else { b'A' };
            data.extend(std::iter::repeat(byte).take(1024));
        }

        if with_insertion {
            let insertion = vec![b'Z'; 512];
            let _ = data.splice(12 * 1024..12 * 1024, insertion);
        }

        data
    }

    fn ingest_pair(config: WindowConfig) -> MetricSnapshot {
        let mut ingestor = Ingestor::new(config);
        let base = Artifact {
            path: "logs/app.log",
            class: "log",
            data: &make_dataset(false),
        };
        let shifted = Artifact {
            path: "logs/app.log",
            class: "log",
            data: &make_dataset(true),
        };

        let _ = ingestor.ingest(base).expect("first ingest");
        ingestor.ingest(shifted).expect("second ingest")
    }

    #[test]
    fn policy_adjusts_windows_and_improves_reuse() {
        let base = WindowConfig {
            min: 4_096,
            avg: 8_192,
            max: 16_384,
        };
        let bounds = WindowBounds {
            min: WindowConfig {
                min: 1_024,
                avg: 2_048,
                max: 4_096,
            },
            max: WindowConfig {
                min: 32_768,
                avg: 65_536,
                max: 98_304,
            },
        };

        let mut policy = WindowPolicy::new(base, bounds, 0.75, 0.35, 0.6, 0.8);

        let baseline_metrics = ingest_pair(base);
        let next_config = policy.observe("log", &baseline_metrics);
        assert!(
            next_config.avg < base.avg,
            "expected shrink toward finer windows"
        );

        let improved_metrics = ingest_pair(next_config);

        assert!(
            improved_metrics.reuse_ratio() > baseline_metrics.reuse_ratio(),
            "smaller windows should reuse more chunks after insertion shift"
        );
        assert!(
            improved_metrics.compression_ratio() <= baseline_metrics.compression_ratio(),
            "smaller windows should not regress compression on repeating data"
        );
    }
}
