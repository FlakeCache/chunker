# 0004 — Storage backend routing lives in the fabric, not the server

**Status:** accepted · **Date:** 2026-07-14 · **Builds on:** [0003](0003-fabric-is-the-data-plane.md)

## Context

ADR-0003 made the fabric the data plane, delivered to the Elixir server as a single
`@behaviour Storage.Backend` adapter over HTTP. Two follow-on questions surfaced while
preparing the prod cutover:

1. **Naming.** The adapter was `FabricBackend` / `FABRIC_ENDPOINT`. "Fabric" collides
   with the SE domain term — `fabrics/build` ("the Build Fabric") is where the node's
   code lives. And the server's existing backends are named for the *storage system*
   they speak (`S3Backend`, `TigrisBackend`, `StorageboxBackend`), never for a domain.
2. **Multiple backends.** If the server needs several storage systems (multiple S3s,
   Storage Box cold, the node) with failover/tiering/weighting, where does that
   *routing* live? The server has hand-rolled `MultiBackend` (primary+fallback) and
   `TieredBackend` (hot+cold), but each leaf reads a single global config — two
   distinct S3s cannot be configured. A "named-instance" refactor of the Elixir
   storage layer (option B) was proposed to fix this.

The inference fabric (`fabrics/inference`) already solved multi-backend routing far
more maturely: a LiteLLM-modeled `gateway-catalog` (named deployments + capabilities
+ pricing + trust), an `adaptive-router` (candidates + routing history), and separate
`gateway-policy` / `gateway-resilience` / `gateway-limits` / `gateway-billing` crates,
with one adapter crate per provider — already spanning **LLM and GPU/compute**
providers. That is the pattern storage (and build runners) would want.

## Decision

**Routing stays out of the Elixir server. The server remains a thin single-backend
client; multi-backend routing is a fabric concern.**

- The storage adapter is named for the system it speaks: **`CasBackend`** (a
  content-addressable store), config `CAS_ENDPOINT` / `CAS_TOKEN`. Symmetric with
  `S3Backend`; the specific deployment is identified by the endpoint, not the type
  name. ("Fabric" is reserved for the `fabrics/build` domain; the Build Fabric
  identity is carried at the deployment level, e.g. instance `build-fabric-1`.)
- We **ship option A** (thin server → one `CasBackend` endpoint) for the cutover.
  Option B (named-instance routing *in Elixir*) is **rejected**: it puts routing in
  the wrong layer.
- When multi-backend routing is genuinely needed (multiple S3s, tiering, weighted
  placement, runners-as-providers), it is built as a **storage gateway in
  `fabrics/build`**, modeled on the inference gateway — a *separate future project*,
  not a blocker for the cutover.

## Why the fabric is the right routing home

**CAS makes placement idempotent.** A blob's identity is its content hash, independent
of location. "Which backend holds blob H" is a clean membership question; a write can
fan out to N backends and a read can fail over across them without any placement
bookkeeping — the hash is the same everywhere. This is precisely the property that
makes request-style routing (as in the inference gateway) sound for storage, and it is
naturally expressed at the CAS/node layer, not in an Elixir dispatcher over
S3-keyed globals.

**Option A is forward-compatible with the gateway.** In the gateway end-state the
Elixir server still points at *one* endpoint (the build-fabric gateway); routing lives
behind it. `CasBackend` is already that thin-client shape. Option B would build
Elixir-side routing that the gateway vision then discards.

## Consequences

- Cutover is unblocked: rename to `CasBackend` (mechanical), rebuild, flip
  `CAS_ENDPOINT` server-wide. Reversible by unsetting it.
- `MultiBackend` / `TieredBackend` stay as-is for now (single-global limitation
  documented, not generalized).
- **Not reusable as-is:** the inference gateway crates are LLM-bound (`llm-provider-core`
  is model/chat-shaped), there is no shared cross-fabric routing substrate
  (`substrate/` is cnpg/garage/nats/valkey), and `adaptive-router` is inference-only.
  A build-fabric storage gateway therefore *mirrors the pattern*; it does not import
  the crates. Extracting a generic provider/catalog/policy/resilience substrate shared
  by inference + build (storage) + compute (runners) is a possible larger play, out of
  scope here.

## Rejected alternatives

- **Option B — named-instance routing in the Elixir storage layer.** Right capability,
  wrong layer; superseded by a fabric-side gateway. Would be thrown away.
- **`FabricBackend` / `BuildFabricBackend` naming.** Names a domain, not a system;
  breaks the `S3Backend`-style convention. Domain belongs in the deployment id.
- **Adopt inference's crates directly for storage.** Not possible without a generic
  extraction; would block the cutover on unrelated refactoring.
