# Goal: FlakeCache distributed storage fabric

> **Fallback tracking artifact** — no host goal-lifecycle tool is exposed in this
> session, so per the goal-setting contract this durable doc stands in for host
> goal mode. Architecture rationale lives in
> [`docs/adr/0001-flakecache-distributed-storage-fabric.md`](../adr/0001-flakecache-distributed-storage-fabric.md).
>
> **Reality check (2026-07-14):** the *current* production FlakeCache is
> `flakecache/server` (Elixir) on K8s + **CNPG + Garage + StorageBox** — **not**
> VectorDrive. This fabric is the *next-gen distributed data-plane* candidate, not
> the shipped system; VectorDrive (graph engine) is a **fabric-only, node-local**
> future for the chunk DAG, gated on its WAL being crash-tested — never the current
> central plane. See
> [`2026-07-14-flakecache-repos-alignment.md`](2026-07-14-flakecache-repos-alignment.md)
> for how this fabric relates to `server` and every other `flakecache/*` repo.

## Goal contract

**End state.** FlakeCache is a distributed, content-addressed, self-healing,
multi-protocol storage fabric: a Rust **data plane** (the nodes *are* the servers
and own the data) plus the existing Elixir **control plane** (auth, tags, GC
coordination, billing, telemetry). It deduplicates Nix / OCI / npm / S3 artifacts
via FastCDC, tiers warm-local ↔ cold-S3/Storage Box, self-heals via leaderless
gossip + rendezvous-hash + CRDT (**no Raft**; CNPG is the only consensus), and
never stores a whole undeduplicated blob or moves bulk bytes through the BEAM —
replacing today's whole-blob-on-S3 behavior.

**Done when.** A swarm of ≥2 nodes serves the Nix binary-cache protocol end to
end — PUT a nar → chunked + deduped + stored across warm/cold tiers → GET from
*any* node (local or peer-fetch) returns byte-exact bytes — GC reclaims
unreferenced chunks, and a **live dedup-ratio measurement on real traffic** shows
chunk dedup beats whole-blob-on-S3 for the workload. Proven by: integration tests
(round-trip + multi-node peer-fetch + GC), `cargo test`/`clippy -D warnings`/`fmt`
green across the workspace, and the measured dedup ratio recorded in
`docs/records/`.

**Scope / boundaries.**
- Primary repo: `flakecache/chunker` (the FlakeCache Rust workspace). Graph-engine
  work lives in `singularity-engine` (data plane) — **no push to shared jj
  bookmarks** without the sanctioned path.
- Dependencies: only battle-tested essentials (`sha3`, `ed25519-dalek`, `redb`,
  `fastcdc`); **std-only** for HTTP/utility; nothing that puts bytes through the
  BEAM. No new heavy deps without an explicit decision.
- **No Raft** anywhere. Data path is conflict-free; the small mutable-metadata
  surface (refcounts/GC/tags/billing) is central Elixir + CNPG.
- Do not deploy to prod or merge feature branches without review.

**Loop.** Build in dependency order, each crate verified (tests, clippy
`-D warnings`, fmt) and pushed before the next: core → backend → swarm → proto →
node → networked. Land the runnable single node first, then the swarm/cold-tier/
control-plane wiring.

**Stop rule.** If a step needs an architecture-changing decision (e.g. trusting
non-atomic graph storage that risks refcount corruption, pulling a heavy dep,
routing bytes through Elixir), or touches prod/shared systems, **stop and report**
— do not widen scope, ship unsafe, or fake success.

## Status (2026-07-14)

**DEPLOYED + VERIFIED.** The single-node fabric is running in k8s (ns
`flakecache-fabric`, image `registry.infra.centralcloud.com/singularity/flakecache-node:dev`,
signing key `flakecache-fabric-1`) and verified against the live pod for *both* modes
singularity-engine uses: **sccache WebDAV** (SE's exact endpoint
`/default/sccache/project/singularity-engine` — cold write + warm hit, verified with
**both sccache 0.12** (SE's pinned version) **and 0.16** (latest, newer opendal)) and
**Nix caching + pushing** (push a store path → substitute it back, signature trusted). The
node speaks the standard Nix HTTP cache protocol + WebDAV; it re-signs narinfos on GET
and stores everything chunked+deduped on a PVC.

Deploy notes: direct `kubectl` (reversible test deploy, **not** GitOps-managed yet).
Not touched: the live Elixir FlakeCache at `cache.flakecache.com/default`. Open decisions:
(1) GitOps-manage the deploy (`deployment/gitops/` Flux) or fold into singularity-engine
per REPO.md — capability → **`fabrics/build/`** (`cache/nix/sccache/oci/chunking` — this
*is* FlakeCache), product/branding → **`products/flakecache/`**; **not** substrate (no such
root exists). (2) prod cutover of `cache.flakecache.com/default` to this node, which needs
the real `default:ESyvaQTi…` key + read/write auth.

**Done — on `main`, pushed (Forgejo + GitHub):**
- `core/chunker` — FastCDC streaming chunker (+ push chunker; Codex-reviewed).
- `core/crypto` — SHAKE-256 + Ed25519 + witness chain (audited `sha3`, not `shake`).
- `core/cas` — content-addressed store + verify + local warm-disk backend (hex inlined).
- `core/meta` — redb DAG + refcounts + mark-and-sweep GC (crash-safe, disk-resident).
- `backend` — memory + warm/cold `TieredBackend` (write-through + read-promote).
- `swarm` — rendezvous-hash placement + content router (local-first, peer-fetch, promote).
- `node` — assembly; **dedup proven** (shared content shares chunks, byte-exact round-trip).
- ADR 0001 + CAS on-disk format spec.

**In review / in progress:**
- `runnable-nix-cache-server` branch — std-only Nix HTTP front (`proto/nix`) + `node-bin`
  binary; verification green; **not pushed** (awaiting review/merge).
- VectorDrive graph-engine WAL crash-safety hardening (singularity-engine) — in progress
  with an adversarial crash-recovery verify queued; graduate the DAG off redb once it
  meets the acceptance criteria (atomic commit, crash replay, crash-recovery tests, page
  mgmt, documented durability, redb parity).

**Remaining milestones (ordered):**
1. Merge the runnable single-node Nix cache; deploy behind Edge Gateway; **measure live dedup**.
2. Cold backend — S3 / Hetzner Storage Box write-through (the one place a client/SigV4 or
   SFTP path is needed).
3. Swarm membership — SWIM gossip (real peer set) + an HTTP `PeerClient` impl.
4. Auth — Ed25519 signed-token verify; Elixir control-plane wiring (tags/GC/billing +
   telemetry-to-Elixir; nodes emit minimal events, no Rust telemetry framework).
5. GC deletion — add `BlobBackend::remove`; wire `collectible_chunks` → delete.
6. More fronts — OCI registry → npm → S3 (hardest, likely `s3s` or hand-rolled SigV4).
7. Graduate the DAG to the VectorDrive graph page-store once its WAL passes acceptance.

## Key decisions (see ADR 0001)
No Raft (CNPG consensus) · nodes-are-servers, data/control split · warm-local + cold-durable,
no whole-blob-on-S3 · embedded on-disk metadata (redb now → VectorDrive graph engine when its
WAL is crash-safe) · FastCDC v2020 committed permanently · VectorDrive canonical, ruvector
frozen donor · DiskANN+RaBitQ vectors as phase-2 similarity · telemetry via central Elixir ·
minimal-dependency (std where possible, audited essentials only).
