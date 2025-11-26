use chunker::{chunking, hashing::HashAlgorithm};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

fn chunk_hashing_benchmarks(c: &mut Criterion) {
    let data = vec![1u8; 2 * 1024 * 1024];
    let mut group = c.benchmark_group("chunking_hashing");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function("sha256", |b| {
        b.iter(|| {
            let chunks = chunking::chunk_data_with_hasher(
                black_box(&data),
                None,
                None,
                None,
                HashAlgorithm::Sha256,
            )
            .expect("chunking should succeed");
            black_box(chunks);
        });
    });

    #[cfg(feature = "blake3")]
    group.bench_function("blake3", |b| {
        b.iter(|| {
            let chunks = chunking::chunk_data_with_hasher(
                black_box(&data),
                None,
                None,
                None,
                HashAlgorithm::Blake3,
            )
            .expect("chunking should succeed");
            black_box(chunks);
        });
    });

    group.finish();
}

criterion_group!(benches, chunk_hashing_benchmarks);
criterion_main!(benches);
