# Latest Benchmark Summary

Generated: `2026-07-08T19:25:55.792628+00:00`
rustc: `rustc 1.96.0 (ac68faa20 2026-05-25)`
cargo: `cargo 1.96.0 (30a34c682 2026-05-25)`
host: `Linux-6.12.91-x86_64-with-glibc2.40`
rustflags: `-C target-cpu=native`

| Benchmark | Duration | Throughput | 95% mean CI |
| --- | ---: | ---: | ---: |
| chunking/chunk_data_eager_10mb | 15.221 ms | 656.99 MiB/s | 15.044 ms - 15.547 ms |
| chunking/chunk_stream_10mb | 36.390 ms | 274.80 MiB/s | 35.692 ms - 36.640 ms |
| chunking/fastcdc_raw_10mb | 4.366 ms | 2.237 GiB/s | 4.341 ms - 4.461 ms |
| compression/zstd_1mb_zeros | 236.735 us | 4.125 GiB/s | 227.243 us - 241.784 us |
| hashing/blake3_1mb | 237.491 us | 4.112 GiB/s | 236.995 us - 246.534 us |
| hashing/sha256_1mb | 614.380 us | 1.590 GiB/s | 613.566 us - 617.518 us |

Source: Criterion JSON under `target/criterion/*/new/`.
Use `just bench` for native CPU benchmarking or `just bench-quick` for a smaller local run.
