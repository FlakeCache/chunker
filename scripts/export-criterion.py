#!/usr/bin/env python3
"""Export Criterion benchmark results into stable repo-local summaries."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def command_output(*command: str) -> str:
    try:
        return subprocess.check_output(command, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def format_duration(ns: float) -> str:
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.3f} ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.3f} us"
    return f"{ns:.3f} ns"


def format_throughput(bytes_per_second: float | None) -> str:
    if bytes_per_second is None:
        return ""

    gib = bytes_per_second / 1024**3
    mib = bytes_per_second / 1024**2

    if gib >= 1.0:
        return f"{gib:.3f} GiB/s"
    return f"{mib:.2f} MiB/s"


def load_benchmark(benchmark_path: Path) -> dict[str, object]:
    estimates_path = benchmark_path.with_name("estimates.json")
    benchmark = json.loads(benchmark_path.read_text())
    estimates = json.loads(estimates_path.read_text())

    mean = estimates["mean"]
    slope = estimates["slope"]
    duration_ns = float(slope["point_estimate"])
    throughput_bytes = benchmark.get("throughput", {}).get("Bytes")
    bytes_per_second = None

    if throughput_bytes is not None:
        bytes_per_second = float(throughput_bytes) / duration_ns * 1_000_000_000.0

    return {
        "id": benchmark["full_id"],
        "group": benchmark.get("group_id"),
        "function": benchmark.get("function_id"),
        "throughput_bytes": throughput_bytes,
        "duration_ns": duration_ns,
        "duration": format_duration(duration_ns),
        "throughput_bytes_per_second": bytes_per_second,
        "throughput": format_throughput(bytes_per_second),
        "mean_ns": float(mean["point_estimate"]),
        "mean_confidence_interval_ns": {
            "lower": float(mean["confidence_interval"]["lower_bound"]),
            "upper": float(mean["confidence_interval"]["upper_bound"]),
            "confidence_level": float(mean["confidence_interval"]["confidence_level"]),
        },
    }


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    rows = payload["benchmarks"]

    lines = [
        "# Latest Benchmark Summary",
        "",
        f"Generated: `{payload['generated_at']}`",
        f"rustc: `{payload['rustc']}`",
        f"cargo: `{payload['cargo']}`",
        f"host: `{payload['host']}`",
        f"rustflags: `{payload['rustflags'] or 'none recorded'}`",
        "",
        "| Benchmark | Duration | Throughput | 95% mean CI |",
        "| --- | ---: | ---: | ---: |",
    ]

    for row in rows:
        ci = row["mean_confidence_interval_ns"]
        lines.append(
            "| {id} | {duration} | {throughput} | {lower} - {upper} |".format(
                id=row["id"],
                duration=row["duration"],
                throughput=row["throughput"],
                lower=format_duration(ci["lower"]),
                upper=format_duration(ci["upper"]),
            )
        )

    lines.extend(
        [
            "",
            "Source: Criterion JSON under `target/criterion/*/new/`.",
            "Use `just bench` for native CPU benchmarking or `just bench-quick` for a smaller local run.",
            "",
        ]
    )

    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--criterion-dir", default="target/criterion")
    parser.add_argument("--output-dir", default="benchmarks")
    args = parser.parse_args()

    criterion_dir = Path(args.criterion_dir)
    output_dir = Path(args.output_dir)
    benchmark_paths = sorted(criterion_dir.glob("**/new/benchmark.json"))

    if not benchmark_paths:
        raise SystemExit(f"no Criterion benchmark results found under {criterion_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rustc": command_output("rustc", "--version"),
        "cargo": command_output("cargo", "--version"),
        "host": platform.platform(),
        "rustflags": os.environ.get("BENCH_RUSTFLAGS") or os.environ.get("RUSTFLAGS") or "",
        "benchmarks": [load_benchmark(path) for path in benchmark_paths],
    }

    (output_dir / "latest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(output_dir / "latest.md", payload)


if __name__ == "__main__":
    main()
