use fastcdc::v2020::FastCDC;
#[cfg(feature = "quickcdc")]
use gearhash::{Hasher as GearHasher, Table as GearTable, DEFAULT_TABLE as DEFAULT_GEAR_TABLE};
use sha2::{Digest, Sha256};

#[derive(Debug, thiserror::Error, Clone, Copy)]
pub enum ChunkingError {
    #[error(
        "bounds_check_failed: offset {offset} + length {length} exceeds data length {data_len}"
    )]
    Bounds {
        data_len: usize,
        offset: usize,
        length: usize,
    },
}

/// A single chunk boundary produced by a strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkBoundary {
    /// Offset into the original buffer.
    pub offset: usize,
    /// Length of the chunk.
    pub length: usize,
}

/// Strategy for producing chunk boundaries.
pub trait ChunkerStrategy: Send + Sync {
    /// Compute chunk boundaries for the provided data.
    fn chunk(&self, data: &[u8]) -> Result<Vec<ChunkBoundary>, ChunkingError>;
}

/// FastCDC configuration.
#[derive(Debug, Clone, Copy)]
pub struct FastCdcConfig {
    pub min_size: usize,
    pub avg_size: usize,
    pub max_size: usize,
}

impl Default for FastCdcConfig {
    fn default() -> Self {
        Self {
            min_size: 16_384,
            avg_size: 65_536,
            max_size: 262_144,
        }
    }
}

#[cfg(feature = "quickcdc")]
/// QuickCDC/RapidCDC-like configuration using a gear mask.
#[derive(Debug, Clone, Copy)]
pub struct QuickCdcConfig {
    pub min_size: usize,
    pub avg_size: usize,
    pub max_size: usize,
    /// Mask used to test for boundaries. Defaults to a value derived from `avg_size` when zero.
    pub mask: u64,
    /// Sliding window size for re-seeding the rolling hash.
    pub window_size: usize,
    /// Optional custom gear table; defaults to the compiled-in table.
    pub table: &'static GearTable,
}

#[cfg(feature = "quickcdc")]
impl Default for QuickCdcConfig {
    fn default() -> Self {
        Self {
            min_size: 16_384,
            avg_size: 65_536,
            max_size: 262_144,
            mask: 0,
            window_size: 64,
            table: &DEFAULT_GEAR_TABLE,
        }
    }
}

/// Two-tier strategy configuration. Coarse boundaries are refined by a fine-grained pass.
#[derive(Debug, Clone)]
pub struct TwoTierConfig {
    pub coarse: Box<ChunkingStrategySelector>,
    pub fine: Box<ChunkingStrategySelector>,
}

impl Default for TwoTierConfig {
    fn default() -> Self {
        Self {
            coarse: Box::new(ChunkingStrategySelector::FastCdc(FastCdcConfig {
                min_size: 32_768,
                avg_size: 131_072,
                max_size: 524_288,
            })),
            fine: Box::new(ChunkingStrategySelector::FastCdc(FastCdcConfig::default())),
        }
    }
}

/// Public selector for chunking strategies.
#[derive(Debug, Clone)]
pub enum ChunkingStrategySelector {
    FastCdc(FastCdcConfig),
    #[cfg(feature = "quickcdc")]
    QuickCdc(QuickCdcConfig),
    TwoTier(TwoTierConfig),
}

impl ChunkingStrategySelector {
    fn build(&self) -> Box<dyn ChunkerStrategy> {
        match self {
            Self::FastCdc(config) => Box::new(FastCdcStrategy { config: *config }),
            #[cfg(feature = "quickcdc")]
            Self::QuickCdc(config) => Box::new(QuickCdcStrategy { config: *config }),
            Self::TwoTier(config) => Box::new(TwoTierStrategy {
                coarse: config.coarse.build(),
                fine: config.fine.build(),
            }),
        }
    }
}

/// Builder-style configuration used by chunk_data.
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    strategy: ChunkingStrategySelector,
}

impl ChunkingConfig {
    pub fn builder() -> ChunkingConfigBuilder {
        ChunkingConfigBuilder {
            strategy: ChunkingStrategySelector::FastCdc(FastCdcConfig::default()),
        }
    }

    pub fn strategy(&self) -> &ChunkingStrategySelector {
        &self.strategy
    }

    pub fn from_strategy(strategy: ChunkingStrategySelector) -> Self {
        Self { strategy }
    }
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            strategy: ChunkingStrategySelector::FastCdc(FastCdcConfig::default()),
        }
    }
}

/// Fluent builder for ChunkingConfig.
#[derive(Debug, Clone)]
pub struct ChunkingConfigBuilder {
    strategy: ChunkingStrategySelector,
}

impl ChunkingConfigBuilder {
    pub fn fastcdc(mut self, config: FastCdcConfig) -> Self {
        self.strategy = ChunkingStrategySelector::FastCdc(config);
        self
    }

    #[cfg(feature = "quickcdc")]
    pub fn quickcdc(mut self, config: QuickCdcConfig) -> Self {
        self.strategy = ChunkingStrategySelector::QuickCdc(config);
        self
    }

    pub fn two_tier(mut self, config: TwoTierConfig) -> Self {
        self.strategy = ChunkingStrategySelector::TwoTier(config);
        self
    }

    pub fn build(self) -> ChunkingConfig {
        ChunkingConfig {
            strategy: self.strategy,
        }
    }
}

/// Validate slice bounds to prevent out-of-bounds access
/// Returns an error if offset + length would exceed `data_len` or overflow
fn validate_slice_bounds(
    data_len: usize,
    offset: usize,
    length: usize,
) -> Result<(), ChunkingError> {
    if offset.checked_add(length).is_none_or(|end| end > data_len) {
        return Err(ChunkingError::Bounds {
            data_len,
            offset,
            length,
        });
    }
    Ok(())
}

/// Chunk data using the configured strategy and return SHA256 hashes for each chunk.
/// Defaults to FastCDC when no configuration is provided.
pub fn chunk_data(
    data: &[u8],
    config: ChunkingConfig,
) -> Result<Vec<(String, usize, usize)>, ChunkingError> {
    let strategy = config.strategy.build();
    let boundaries = strategy.chunk(data)?;

    let mut chunks = Vec::with_capacity(boundaries.len());

    for chunk in boundaries {
        validate_slice_bounds(data.len(), chunk.offset, chunk.length)?;

        let mut hasher = Sha256::new();
        hasher.update(&data[chunk.offset..chunk.offset + chunk.length]);
        let hash_hex = hex::encode(hasher.finalize());

        chunks.push((hash_hex, chunk.offset, chunk.length));
    }

    Ok(chunks)
}

/// Convenience wrapper that builds a FastCDC configuration from explicit sizes.
pub fn chunk_data_with_sizes(
    data: &[u8],
    min_size: Option<usize>,
    avg_size: Option<usize>,
    max_size: Option<usize>,
) -> Result<Vec<(String, usize, usize)>, ChunkingError> {
    let fastcdc_config = FastCdcConfig {
        min_size: min_size.unwrap_or(16_384),
        avg_size: avg_size.unwrap_or(65_536),
        max_size: max_size.unwrap_or(262_144),
    };

    let config = ChunkingConfig::builder().fastcdc(fastcdc_config).build();
    chunk_data(data, config)
}

struct FastCdcStrategy {
    config: FastCdcConfig,
}

impl ChunkerStrategy for FastCdcStrategy {
    fn chunk(&self, data: &[u8]) -> Result<Vec<ChunkBoundary>, ChunkingError> {
        #[allow(clippy::cast_possible_truncation)]
        let min = self.config.min_size as u32;
        #[allow(clippy::cast_possible_truncation)]
        let avg = self.config.avg_size as u32;
        #[allow(clippy::cast_possible_truncation)]
        let max = self.config.max_size as u32;

        let chunker = FastCDC::new(data, min, avg, max);
        let mut chunks = Vec::new();
        for chunk in chunker {
            validate_slice_bounds(data.len(), chunk.offset, chunk.length)?;
            chunks.push(ChunkBoundary {
                offset: chunk.offset,
                length: chunk.length,
            });
        }

        Ok(chunks)
    }
}

#[cfg(feature = "quickcdc")]
struct QuickCdcStrategy {
    config: QuickCdcConfig,
}

#[cfg(feature = "quickcdc")]
impl QuickCdcStrategy {
    fn derive_mask(&self) -> u64 {
        if self.config.mask != 0 {
            self.config.mask
        } else {
            let avg_power = self.config.avg_size.next_power_of_two();
            (avg_power as u64).saturating_sub(1)
        }
    }

    fn find_boundary(&self, data: &[u8], start: usize) -> usize {
        let remaining = data.len() - start;
        if remaining <= self.config.min_size {
            return data.len();
        }

        let min_end = (start + self.config.min_size).min(data.len());
        let max_end = (start + self.config.max_size).min(data.len());
        let mask = self.derive_mask();

        let mut idx = min_end;
        let mut hasher = GearHasher::new(self.config.table);
        let window_seed_start = idx.saturating_sub(self.config.window_size);
        hasher.update(&data[window_seed_start..idx]);

        while idx < max_end {
            hasher.update(&data[idx..idx + 1]);
            if hasher.is_match(mask) {
                return idx + 1;
            }

            idx += 1;

            if self.config.window_size > 0 && (idx - start) % self.config.window_size == 0 {
                let window_start = idx.saturating_sub(self.config.window_size);
                let mut window_hasher = GearHasher::new(self.config.table);
                window_hasher.update(&data[window_start..idx]);
                hasher = window_hasher;
            }
        }

        max_end
    }
}

#[cfg(feature = "quickcdc")]
impl ChunkerStrategy for QuickCdcStrategy {
    fn chunk(&self, data: &[u8]) -> Result<Vec<ChunkBoundary>, ChunkingError> {
        let mut chunks = Vec::new();
        let mut offset = 0;

        while offset < data.len() {
            let end = self.find_boundary(data, offset);
            let length = end - offset;
            validate_slice_bounds(data.len(), offset, length)?;
            chunks.push(ChunkBoundary { offset, length });
            offset = end;
        }

        Ok(chunks)
    }
}

struct TwoTierStrategy {
    coarse: Box<dyn ChunkerStrategy>,
    fine: Box<dyn ChunkerStrategy>,
}

impl ChunkerStrategy for TwoTierStrategy {
    fn chunk(&self, data: &[u8]) -> Result<Vec<ChunkBoundary>, ChunkingError> {
        let coarse_chunks = self.coarse.chunk(data)?;
        let mut refined = Vec::new();

        for coarse in coarse_chunks {
            let slice_end = coarse.offset + coarse.length;
            validate_slice_bounds(data.len(), coarse.offset, coarse.length)?;
            let slice = &data[coarse.offset..slice_end];
            let fine_chunks = self.fine.chunk(slice)?;
            for fine in fine_chunks {
                validate_slice_bounds(slice.len(), fine.offset, fine.length)?;
                refined.push(ChunkBoundary {
                    offset: coarse.offset + fine.offset,
                    length: fine.length,
                });
            }
        }

        Ok(refined)
    }
}
