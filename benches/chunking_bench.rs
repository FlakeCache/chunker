use chunker::chunking::{
    chunk_boundaries_with_strategy, chunk_data_with_strategy, ChunkingStrategyKind,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn sample_data(bytes: usize) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(0x5EED_F00Du64);
    let mut data = vec![0u8; bytes];
    rng.fill(&mut data[..]);
    data
}

fn boundary_benchmarks(c: &mut Criterion) {
    let data = sample_data(8 * 1024 * 1024); // 8 MiB
    let min = 16_384;
    let avg = 65_536;
    let max = 262_144;

    c.bench_function("fastcdc_boundaries", |b| {
        b.iter(|| {
            let chunks = chunk_boundaries_with_strategy(
                black_box(&data),
                Some(min),
                Some(avg),
                Some(max),
                ChunkingStrategyKind::FastCdc,
            )
            .unwrap();
            black_box(chunks);
        })
    });

    #[cfg(feature = "quickcdc")]
    {
        c.bench_function("quickcdc_boundaries", |b| {
            b.iter(|| {
                let chunks = chunk_boundaries_with_strategy(
                    black_box(&data),
                    Some(min),
                    Some(avg),
                    Some(max),
                    ChunkingStrategyKind::QuickCdc,
                )
                .unwrap();
                black_box(chunks);
            })
        });
    }
}

fn hashing_benchmarks(c: &mut Criterion) {
    let data = sample_data(8 * 1024 * 1024); // 8 MiB
    let min = 16_384;
    let avg = 65_536;
    let max = 262_144;

    c.bench_function("fastcdc_hashing", |b| {
        b.iter(|| {
            let chunks = chunk_data_with_strategy(
                black_box(&data),
                Some(min),
                Some(avg),
                Some(max),
                ChunkingStrategyKind::FastCdc,
            )
            .unwrap();
            black_box(chunks);
        })
    });

    #[cfg(feature = "quickcdc")]
    {
        c.bench_function("quickcdc_hashing", |b| {
            b.iter(|| {
                let chunks = chunk_data_with_strategy(
                    black_box(&data),
                    Some(min),
                    Some(avg),
                    Some(max),
                    ChunkingStrategyKind::QuickCdc,
                )
                .unwrap();
                black_box(chunks);
            })
        });
    }
}

criterion_group!(chunking, boundary_benchmarks, hashing_benchmarks);
criterion_main!(chunking);
