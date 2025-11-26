#[cfg(feature = "quickcdc")]
use chunker::chunking::QuickCdcConfig;
use chunker::chunking::{
    chunk_data, ChunkingConfig, ChunkingStrategySelector, FastCdcConfig, TwoTierConfig,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn make_data() -> Vec<u8> {
    // Deterministic pseudo-random data to keep benchmarks stable.
    let mut data = Vec::with_capacity(2 * 1024 * 1024);
    let mut state: u32 = 0x1234_5678;
    for _ in 0..data.capacity() {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        data.push((state >> 16) as u8);
    }
    data
}

fn bench_fastcdc(c: &mut Criterion, data: &[u8]) {
    let config = ChunkingConfig::builder()
        .fastcdc(FastCdcConfig::default())
        .build();
    let mut group = c.benchmark_group("chunking_fastcdc");
    group.bench_function(BenchmarkId::from_parameter("default"), |b| {
        b.iter(|| chunk_data(black_box(data), config.clone()))
    });
    group.finish();
}

#[cfg(feature = "quickcdc")]
fn bench_quickcdc(c: &mut Criterion, data: &[u8]) {
    let config = ChunkingConfig::builder()
        .quickcdc(QuickCdcConfig::default())
        .build();
    let mut group = c.benchmark_group("chunking_quickcdc");
    group.bench_function(BenchmarkId::from_parameter("default"), |b| {
        b.iter(|| chunk_data(black_box(data), config.clone()))
    });
    group.finish();
}

fn bench_two_tier(c: &mut Criterion, data: &[u8]) {
    #[cfg(feature = "quickcdc")]
    let coarse = ChunkingStrategySelector::QuickCdc(QuickCdcConfig {
        min_size: 32_768,
        avg_size: 131_072,
        max_size: 524_288,
        ..QuickCdcConfig::default()
    });

    #[cfg(not(feature = "quickcdc"))]
    let coarse = ChunkingStrategySelector::FastCdc(FastCdcConfig {
        min_size: 32_768,
        avg_size: 131_072,
        max_size: 524_288,
    });

    let fine = ChunkingStrategySelector::FastCdc(FastCdcConfig::default());
    let config = ChunkingConfig::from_strategy(ChunkingStrategySelector::TwoTier(TwoTierConfig {
        coarse: Box::new(coarse),
        fine: Box::new(fine),
    }));

    let mut group = c.benchmark_group("chunking_two_tier");
    group.bench_function(BenchmarkId::from_parameter("coarse+fine"), |b| {
        b.iter(|| chunk_data(black_box(data), config.clone()))
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let data = make_data();
    bench_fastcdc(c, &data);
    #[cfg(feature = "quickcdc")]
    bench_quickcdc(c, &data);
    bench_two_tier(c, &data);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
