use chunker::{compression, hashing};
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};

fn benchmark_hashing(c: &mut Criterion) {
    let mut group = c.benchmark_group("hashing");
    let size = 1024 * 1024; // 1MB
    let mut data = vec![0u8; size];
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    rng.fill_bytes(&mut data);

    let _ = group.throughput(Throughput::Bytes(size as u64));

    let _ = group.bench_function("sha256", |b| {
        b.iter(|| {
            let _ = black_box(hashing::sha256_hash(black_box(&data)));
        })
    });

    let _ = group.bench_function("blake3", |b| {
        b.iter(|| {
            let _ = black_box(hashing::blake3_hash(black_box(&data)));
        })
    });

    group.finish();
}

fn benchmark_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression");
    let size = 1024 * 1024; // 1MB
    let mut data = vec![0u8; size];
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    rng.fill_bytes(&mut data);

    let _ = group.throughput(Throughput::Bytes(size as u64));

    let _ = group.bench_function("zstd_level_3", |b| {
        b.iter(|| {
            let _ = black_box(compression::compress_zstd(black_box(&data), Some(3)));
        });
    });

    let _ = group.bench_function("lz4", |b| {
        b.iter(|| {
            let _ = black_box(compression::compress_lz4(black_box(&data)));
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_hashing, benchmark_compression);
criterion_main!(benches);
