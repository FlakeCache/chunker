#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::panic)]

use chunker::{
    chunking::{ChunkStream, HashAlgorithm, chunk_data_with_hash, chunk_descriptors_with_hash},
    compression::compress_zstd,
    hashing::{blake3_hash, sha256_hash},
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fastcdc::v2020::FastCDC;
use std::hint::black_box;
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

    let _ = group.bench_function("chunk_data_eager_10mb_sha256", |b| {
        b.iter(|| {
            let _ = chunker::chunking::chunk_data(black_box(&data), None, None, None);
        });
    });

    let _ = group.bench_function("chunk_data_eager_10mb_blake3", |b| {
        b.iter(|| {
            let _ = chunk_data_with_hash(black_box(&data), None, None, None, HashAlgorithm::Blake3);
        });
    });

    let _ = group.bench_function("chunk_descriptors_10mb_sha256", |b| {
        b.iter(|| {
            let _ = chunk_descriptors_with_hash(
                black_box(&data),
                None,
                None,
                None,
                HashAlgorithm::Sha256,
            );
        });
    });

    let _ = group.bench_function("chunk_descriptors_10mb_blake3", |b| {
        b.iter(|| {
            let _ = chunk_descriptors_with_hash(
                black_box(&data),
                None,
                None,
                None,
                HashAlgorithm::Blake3,
            );
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
    let _ = group.bench_function("blake3_1mb", |b| {
        b.iter(|| {
            let _ = blake3_hash(black_box(&data));
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
