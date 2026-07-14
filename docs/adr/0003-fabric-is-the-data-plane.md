# ADR 0003 — The fabric is FlakeCache's data plane (as a `server` Storage.Backend)

Status: accepted · 2026-07-14 · resolves the open A/B decision in
[`../plans/2026-07-14-flakecache-repos-alignment.md`](../plans/2026-07-14-flakecache-repos-alignment.md)

## Decision

Adopt **(A)**: the Rust `chunker/node` fabric is FlakeCache's next-gen **data plane**
— content-defined dedup, self-healing swarm, warm-local + cold-durable tiering.

It is delivered **not as a rewrite of `server`** but as a new **`FabricBackend`
implementing `server`'s existing `@behaviour FlakecacheApp.Storage.Backend`** (beside
`S3Backend` / `LocalBackend` / `StorageboxBackend` / `TieredBackend`). `server` keeps
**all** of: the protocols (Nix / sccache / GHA / OCI / flake-registry), auth / tenancy
/ billing, per-cache zero-knowledge signing, and GC / retention policy. **The fabric
owns only the bytes.**

## Why this shape (not a monolith rewrite, not a big-bang cutover)

- `server` is battle-tested and owns the hard product surface — duplicating it is
  waste and risk.
- `server`'s storage is **already** a pluggable `@behaviour`; the fabric slots in as
  one more backend — **reversible** (swap back to `S3Backend`), **canary-able** per
  cache.
- The fabric's real advantage over Garage-whole-blob is **chunk dedup + self-heal +
  warm/cold tiering** — a storage-tier improvement, which is exactly what a
  `Storage.Backend` is.
- Preserves the data/control split (Rust owns bytes, Elixir owns control) proven this
  session.

## FabricBackend contract (`server` → fabric node)

`server`'s `Storage.Backend` callbacks map onto the fabric node's HTTP API:

| Backend callback | Fabric HTTP |
|---|---|
| `put(key, blob)` | `PUT /<key>` (fabric chunks + dedups) |
| `get(key)` | `GET /<key>` (reassembled, byte-exact) |
| `delete(key)` | `DELETE /<key>` |
| `exists?(key)` | `HEAD /<key>` |

`list` is not needed (GC is `server`-driven). **Auth** (fabric verifies a
`server`-minted signed token on write/read) is **required before any prod traffic** —
the node is currently unauthenticated.

## Phased rollout (reversible at every step)

0. **DONE** — fabric node deployed (`flakecache-fabric` ns) + verified for Nix and
   sccache (0.12 & 0.16).
1. **Cold durability** — fabric cold backend (Garage S3 or Hetzner StorageBox
   write-through) so blobs survive node/PVC loss. *[chunker]*
2. **Auth** — fabric verifies `server`-minted signed tokens on `PUT`/`GET`. *[chunker]*
3. **`FabricBackend` adapter** in `server` (`Storage.Backend` → HTTP to the fabric). *[server]*
4. **Canary** — route **one** non-critical cache's storage through `FabricBackend`;
   measure dedup ratio + latency + reliability vs `S3Backend`; auto-fallback on error.
5. **Broaden / cutover** — expand cache-by-cache; the `cache.flakecache.com/default`
   cutover uses the real `default` key.

## Consequences

- No `server` rewrite; the fabric is additive and reversible.
- The fabric's remaining milestones (cold backend, auth, swarm gossip, GC-delete)
  become the **phase gates** above — they're no longer open-ended.
- VectorDrive graph-engine graduation for node-local metadata stays independent, gated
  on its WAL being crash-tested (`redb` until then).
- `sccache` and the GHA cache flow through the same `FabricBackend` once `server`
  routes those stores to it — no separate integration.
