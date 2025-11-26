use chunker::chunking::{chunk_data_with_backend, ChunkingBackend};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::RngCore;

fn large_random_blob(size: usize) -> Vec<u8> {
    let mut data = vec![0u8; size];
    rand::thread_rng().fill_bytes(&mut data);
    data
}

fn bench_chunking(c: &mut Criterion) {
    let data = large_random_blob(32 * 1024 * 1024); // 32 MiB
    let mut group = c.benchmark_group("chunking-throughput");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function("cpu_fastcdc", |b| {
        b.iter(|| {
            let result =
                chunk_data_with_backend(ChunkingBackend::Cpu, black_box(&data), None, None, None);
            assert!(result.is_ok());
        });
    });

    #[cfg(feature = "gpu")]
    {
        group.bench_function("gpu_auto", |b| {
            b.iter(|| {
                let result = chunk_data_with_backend(
                    ChunkingBackend::Auto,
                    black_box(&data),
                    None,
                    None,
                    None,
                );
                assert!(result.is_ok());
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_chunking);
criterion_main!(benches);
