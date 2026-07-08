# Latest Benchmark Summary

Generated: `2026-07-08T19:33:05.544747+00:00`
rustc: `rustc 1.96.0 (ac68faa20 2026-05-25)`
cargo: `cargo 1.96.0 (30a34c682 2026-05-25)`
host: `Linux-6.12.91-x86_64-with-glibc2.40`
rustflags: `-C target-cpu=native`

| Benchmark | Duration | Throughput | 95% mean CI |
| --- | ---: | ---: | ---: |
| chunking/chunk_data_eager_10mb | 15.774 ms | 633.95 MiB/s | 15.496 ms - 15.848 ms |
| chunking/chunk_stream_10mb | 35.431 ms | 282.24 MiB/s | 35.439 ms - 36.534 ms |
| chunking/fastcdc_raw_10mb | 4.292 ms | 2.275 GiB/s | 4.275 ms - 4.368 ms |
| compression/zstd_1mb_zeros | 222.822 us | 4.383 GiB/s | 218.783 us - 224.516 us |
| hashing/blake3_1mb | 242.067 us | 4.034 GiB/s | 236.303 us - 243.698 us |
| hashing/sha256_1mb | 605.667 us | 1.612 GiB/s | 604.927 us - 605.996 us |

Source: Criterion JSON under `target/criterion/*/new/`.
Use `just bench` for native CPU benchmarking or `just bench-quick` for a smaller local run.
