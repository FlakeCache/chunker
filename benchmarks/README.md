# Benchmarks

`latest.json` and `latest.md` are exported from Criterion output with:

```bash
just bench
```

Use `just bench-quick` for a shorter local run. Criterion keeps detailed HTML
and sample data under `target/criterion/`; this directory stores the stable
summary that is safe to review in git.

See `DEPENDENCY_POLICY.md` before accepting hashing or chunking dependency
upgrades.
