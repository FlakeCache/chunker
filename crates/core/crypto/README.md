# flakecache-crypto

`no_std` cryptographic primitives for FlakeCache's content-addressed storage fabric.

- **`hash`** — SHAKE-256 content hashing for chunk/manifest identifiers.
- **`sign`** — Ed25519 sign/verify (Nix binary caches are cryptographically signed).
- **`witness`** — a hash-chained witness log giving tamper-evident integrity over the store's writes.

Derived from the `rvf-crypto` design; re-scoped to content-hash-keyed use.
Default build is `no_std` + `alloc`; enable the `std` feature for std integration.
