use chunker::{
    compression::compress_zstd,
    hashing::sha256_hash,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use fastcdc::v2020::FastCDC;

fn benchmark_chunking(c: &mut Criterion) {
    let size = 10 * 1024 * 1024; // 10 MB
    let data = vec![0u8; size];
    
    let mut group = c.benchmark_group("chunking");
    let _ = group.throughput(Throughput::Bytes(size as u64));
    let _ = group.bench_function("fastcdc_10mb", |b| {
        b.iter(|| {
            let chunks = FastCDC::new(black_box(&data), 1024, 4096, 16384);
            for _ in chunks {}
        });
    });
    group.finish();
}

fn benchmark_hashing(c: &mut Criterion) {
    let size = 1024 * 1024; // 1 MB
    let data = vec![0u8; size];

    let mut group = c.benchmark_group("hashing");
    let _ = group.throughput(Throughput::Bytes(size as u64));
    let _ = group.bench_function("sha256_1mb", |b| {
        b.iter(|| {
            let _ = sha256_hash(black_box(&data));
        });
    });
    group.finish();
}

fn benchmark_compression(c: &mut Criterion) {
    let size = 1024 * 1024; // 1 MB
    let data = vec![0u8; size]; // All zeros, highly compressible

    let mut group = c.benchmark_group("compression");
    let _ = group.throughput(Throughput::Bytes(size as u64));
    let _ = group.bench_function("zstd_1mb_zeros", |b| {
        b.iter(|| {
            assert!(
                compress_zstd(black_box(&data), Some(3)).is_ok(),
                "zstd compression failed during benchmark"
            );
        });
    });
    group.finish();
}

criterion_group!(benches, benchmark_chunking, benchmark_hashing, benchmark_compression);
criterion_main!(benches);
