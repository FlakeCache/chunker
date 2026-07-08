# Latest Benchmark Summary

Generated: `2026-07-08T19:55:28.767247+00:00`
rustc: `rustc 1.96.0 (ac68faa20 2026-05-25)`
cargo: `cargo 1.96.0 (30a34c682 2026-05-25)`
host: `Linux-6.12.91-x86_64-with-glibc2.40`
rustflags: `-C target-cpu=native`

| Benchmark | Duration | Throughput | 95% mean CI |
| --- | ---: | ---: | ---: |
| chunking/chunk_data_eager_10mb_blake3 | 7.685 ms | 1.271 GiB/s | 7.588 ms - 7.856 ms |
| chunking/chunk_data_eager_10mb_sha256 | 9.254 ms | 1.055 GiB/s | 9.260 ms - 9.530 ms |
| chunking/chunk_descriptors_10mb_blake3 | 6.611 ms | 1.477 GiB/s | 6.550 ms - 6.777 ms |
| chunking/chunk_descriptors_10mb_sha256 | 7.996 ms | 1.221 GiB/s | 7.832 ms - 8.006 ms |
| chunking/chunk_stream_10mb | 14.021 ms | 713.23 MiB/s | 13.624 ms - 14.068 ms |
| chunking/fastcdc_raw_10mb | 5.096 ms | 1.916 GiB/s | 5.029 ms - 5.137 ms |
| compression/zstd_1mb_zeros | 236.087 us | 4.136 GiB/s | 232.338 us - 237.461 us |
| hashing/blake3_1mb | 238.335 us | 4.097 GiB/s | 234.656 us - 240.656 us |
| hashing/sha256_1mb | 616.246 us | 1.585 GiB/s | 610.843 us - 616.721 us |

Source: Criterion JSON under `target/criterion/*/new/`.
Use `just bench` for native CPU benchmarking or `just bench-quick` for a smaller local run.
