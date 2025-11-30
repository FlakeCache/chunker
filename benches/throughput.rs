#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::panic)]

use chunker::{chunking::ChunkStream, compression::compress_zstd, hashing::sha256_hash};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use fastcdc::v2020::FastCDC;
use std::io::Cursor;

fn benchmark_chunking(c: &mut Criterion) {
    let size = 10 * 1024 * 1024; // 10 MB
    let data = vec![0u8; size];

    let mut group = c.benchmark_group("chunking");
    let _ = group.throughput(Throughput::Bytes(size as u64));

    let _ = group.bench_function("fastcdc_raw_10mb", |b| {
        b.iter(|| {
            let chunks = FastCDC::new(black_box(&data), 256 * 1024, 1024 * 1024, 4 * 1024 * 1024);
            for _ in chunks {}
        });
    });

    let _ = group.bench_function("chunk_stream_10mb", |b| {
        b.iter(|| {
            let cursor = Cursor::new(black_box(&data));
            let stream = ChunkStream::new(cursor, None, None, None).unwrap();
            for chunk in stream {
                let _ = black_box(chunk.unwrap());
            }
        });
    });

    let _ = group.bench_function("chunk_data_eager_10mb", |b| {
        b.iter(|| {
            let _ = chunker::chunking::chunk_data(black_box(&data), None, None, None);
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
            let _ = black_box(compress_zstd(black_box(&data), Some(3)));
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    benchmark_chunking,
    benchmark_hashing,
    benchmark_compression
);
criterion_main!(benches);
