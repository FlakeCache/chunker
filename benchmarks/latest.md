# Latest Benchmark Summary

Generated: `2026-07-08T19:41:21.129928+00:00`
rustc: `rustc 1.96.0 (ac68faa20 2026-05-25)`
cargo: `cargo 1.96.0 (30a34c682 2026-05-25)`
host: `Linux-6.12.91-x86_64-with-glibc2.40`
rustflags: `-C target-cpu=native`

| Benchmark | Duration | Throughput | 95% mean CI |
| --- | ---: | ---: | ---: |
| chunking/chunk_data_eager_10mb | 16.876 ms | 592.54 MiB/s | 16.626 ms - 17.628 ms |
| chunking/chunk_stream_10mb | 39.546 ms | 252.87 MiB/s | 36.419 ms - 40.262 ms |
| chunking/fastcdc_raw_10mb | 5.273 ms | 1.852 GiB/s | 5.090 ms - 5.262 ms |
| compression/zstd_1mb_zeros | 220.717 us | 4.424 GiB/s | 218.528 us - 236.122 us |
| hashing/blake3_1mb | 235.774 us | 4.142 GiB/s | 232.457 us - 237.781 us |
| hashing/sha256_1mb | 620.271 us | 1.574 GiB/s | 610.772 us - 619.433 us |

Source: Criterion JSON under `target/criterion/*/new/`.
Use `just bench` for native CPU benchmarking or `just bench-quick` for a smaller local run.
