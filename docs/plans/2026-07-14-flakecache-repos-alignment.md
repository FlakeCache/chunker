# FlakeCache — cross-repo architecture alignment

Date: 2026-07-14 · Scope: all `flakecache/*` repos · Status: alignment plan

## Ground truth (read this first)

The **current production FlakeCache is `flakecache/server`** (Elixir/Phoenix) —
it serves `cache.flakecache.com` on **Kubernetes** (`flakecache` ns StatefulSet) +
**CNPG Postgres** (metadata) + **Garage S3** (`centralcloud-flakecache`, hot blobs) +
**Hetzner StorageBox** (cold tier, `TieredBackend`). It terminates *every* protocol:
Nix narinfo/NAR, **sccache WebDAV** (`/:cache/sccache`, prod-smoke passing),
GitHub/Forgejo Actions cache, OCI registry (built, disabled), flake registry — plus
auth/SSO, Stripe billing (flag-off), and org/tenant management.

`server/docs/architecture/CURRENT.md` + `PHASES.md` (2026-07-09, self-declared
ground truth) are **authoritative for current state**. This doc defers to them and
adds: (1) the cross-repo role + SE-taxonomy map, (2) the Rust-fabric direction, and
(3) corrections to over-eager claims made in this session's chunker ADRs.

## Repos → role → placement

Placement uses singularity-engine's `REPO.md` taxonomy (`fabrics/ domains/ platform/
engine/ products/ deployment/` — **no `substrate/`**).

| Repo | Role (verified) | SE placement |
|---|---|---|
| **server** | Multi-protocol cache **+** product control plane (Nix/sccache/GHA/OCI/flake-reg, auth/billing/tenancy). Elixir; K8s+CNPG+Garage+StorageBox; **zero-knowledge per-cache signing**. | **`products/flakecache/`** (the sellable product) composing **`domains/billing`**, **`domains/identity/organizations`**; its protocol serving is the product's data serving today. |
| **chunker** | (a) Rust **primitives lib** — FastCDC, hashing, compression, Ed25519 — used as *server's Rustler NIF* (`native/chunker` symlink) and *cli's* crate; (b) **NEW fabric node** (`crates/node`+`proto/nix`, this session) = a standalone Rust cache server. | primitives → **`fabrics/build/chunking`** (crypto → `platform` or here); fabric node → **`fabrics/build/{cache,nix,sccache}`**. |
| **cli** | Rust client: push/pull/warm, OAuth, **client-side (zero-knowledge) signing**. `API_MAPPING.md` = the client↔server wire contract. | **`products/flakecache/`** (client). |
| **runner** | Elixir "flakebuilder" — CI **execution** substrate (Firecracker microVMs on k8s). Consumes server's cache via HTTP proxy. Dev, not prod (prod = `forgejo-runner` bridge). | **`fabrics/compute/runners`** — *not* `fabrics/build`. |
| **actions-cache** | Rust GHA-cache **edge sidecar** (Forgejo HMAC auth + reverse-proxy → server; unused standalone S3 mode). | **`fabrics/build/cache`** (edge) — proxy nature is `platform`-adjacent. |
| **builder** | Nix wrapper for **distributed compiler** (`sccache-dist`); extraction target *from* runner; feeds artifacts to server's WebDAV. | **`fabrics/build/{scheduling,sccache}`**. |
| **crate-index / nix-installer / releases** | Cargo sparse index for chunker; CI Nix-installer action; CLI binary hosting. | distribution → **`products/flakecache/`** + **`deployment/tools`**. |
| **devops** | Infra/CI conventions (`FORGEJO_WORKFLOWS.md` = canonical CI SoT), k8s ARC runners, Fly-era DB tooling. | **`deployment/`**. |
| **tailscale** | Genuine **vendored upstream fork** (tsnet networking); only local change is a `nixConfig` pointing builds at the cache. | **`code/vendors/`** (devops already flags this) — not an owned component. |

## Shared architecture (stamp into every plan)

- **Data plane (current):** CNPG (metadata) + Garage S3 (hot) + StorageBox (cold),
  via server's `@behaviour` `TieredBackend`. **Not VectorDrive** — correcting an
  earlier chunker-doc claim. pgvector is explicitly rejected today (a later
  flake-semantic-search epic reconsiders it; unshipped).
- **Trust:** per-cache Ed25519 keys, **zero-knowledge signing** (pusher holds the
  secret; `caches` has `signing_public_key`, no secret column; server stores
  pusher-supplied sigs verbatim) — **already in production in server**, matching
  cachix. The chunker fabric node adds an optional **managed server-side re-sign**
  mode (ADR 0002 "mode B") — its value is exactly fixing *unsigned* CI pushes
  (the observed narinfo break is the pusher not signing, not the server).
- **No Raft.** Central authority = CNPG.
- **Deploy:** K8s + Flux; wiring in `/srv/infra/clusters/default/tenants/shared/apps/flakecache/`
  (migrating into `deployment/gitops/` per REPO.md). **Fly.io is retired** ("do not revive").

## The one real decision — fabric vs. server data path — RESOLVED (A, phased)

**Decision (2026-07-14, [ADR 0003](../adr/0003-fabric-is-the-data-plane.md)):** adopt
**(A)** — the fabric is the data plane — but delivered as a **`FabricBackend`
implementing `server`'s existing `@behaviour Storage.Backend`**, not a rewrite.
`server` keeps every protocol + auth + billing + signing; the fabric owns only the
bytes (dedup + self-heal + tiering), reversible to `S3Backend`, canary-able per cache.
Phased: cold backend → auth → `FabricBackend` adapter → one-cache canary → broaden.
The rest of this section is the original framing that led there.


`server` already serves Nix + sccache + GHA cache + warm/cold tiering **in prod**.
The `chunker/node` fabric (Rust swarm, content-defined dedup, self-heal) is a
**second data-path implementation**. This is a deliberate fork in the road, not doc
drift:

- **(A) Fabric = next-gen distributed data plane.** server keeps protocol +
  tenancy + billing + signing; the byte storage migrates onto the Rust swarm
  (dedup, self-healing, warm-local + cold-S3). Matches the data/control split and
  the "self-healing swarm" vision from this session.
- **(B) Fabric = R&D / a specific edge tier.** server stays the monolith; the
  fabric proves ideas or serves one edge case.

`PHASES.md`'s edge strategy (actions-cache sidecar vs. regional replica) is
explicitly **undecided** — the fabric is a *third* candidate for that layer. The
`flakecache-fabric` deploy stood up this session is a **verified proof-of-concept of
(A)'s data path** (real `nix` push/substitute + sccache 0.12 & 0.16 hits), **not** a
replacement of server. Pick A or B deliberately.

## Doc-drift to reconcile (per repo)

- **server:** Fly.io docs (`CLAUDE.md`, `DEPLOYMENT.md`, `DEPLOY_FLY.md`, `STATUS.md`)
  are stale — K8s is ground truth. `SINGULARITY_ENGINE_*.md` are aspirational/out of
  scope for what's built. `DISTRIBUTED_CI_CACHE.md` already marked superseded.
- **chunker:** fix the "central plane on VectorDrive" line + acknowledge server's
  existing zero-knowledge signing as the reference (done alongside this doc).
- **runner / actions-cache / builder:** already point to `server/docs/architecture`
  as SoT; add the taxonomy placement above to their plans.
- **fabric node-local metadata:** `redb` (crash-safe) now → VectorDrive graph engine
  when its WAL is crash-tested. Fabric-only; server stays on CNPG.

## Next actions

1. ✅ **Decided A** (fabric = data plane via `server` `Storage.Backend` adapter) — [ADR 0003](../adr/0003-fabric-is-the-data-plane.md).
2. **Phase 1** — fabric **cold backend** (Garage S3 / StorageBox write-through) so blobs are durable. *[chunker, next]*
3. **Phase 2** — fabric **auth** (verify `server`-minted signed tokens).
4. **Phase 3** — `FabricBackend` adapter in `server`; **Phase 4** — one-cache canary with `S3Backend` fallback; **Phase 5** — broaden + `default` cutover.
5. ✅ Taxonomy placement landed in each repo (`ALIGNMENT.md`).
