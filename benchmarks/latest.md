# Latest Benchmark Summary

Generated: `2026-07-08T18:28:49.375569+00:00`
rustc: `rustc 1.96.0 (ac68faa20 2026-05-25)`
cargo: `cargo 1.96.0 (30a34c682 2026-05-25)`
host: `Linux-6.12.91-x86_64-with-glibc2.40`
rustflags: `-C target-cpu=native`

| Benchmark | Duration | Throughput | 95% mean CI |
| --- | ---: | ---: | ---: |
| chunking/chunk_data_eager_10mb | 16.098 ms | 621.19 MiB/s | 15.751 ms - 16.307 ms |
| chunking/chunk_stream_10mb | 36.057 ms | 277.34 MiB/s | 35.878 ms - 36.179 ms |
| chunking/fastcdc_raw_10mb | 4.373 ms | 2.233 GiB/s | 4.340 ms - 4.556 ms |
| compression/zstd_1mb_zeros | 249.495 us | 3.914 GiB/s | 246.407 us - 253.985 us |
| hashing/sha256_1mb | 607.903 us | 1.606 GiB/s | 605.830 us - 608.135 us |

Source: Criterion JSON under `target/criterion/*/new/`.
Use `just bench` for native CPU benchmarking or `just bench-quick` for a smaller local run.
