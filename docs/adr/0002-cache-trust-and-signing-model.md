# ADR 0002 — Cache trust & signing model

Status: accepted · 2026-07-14 · supersedes the "own key per node" sketch in discussion

## Context

FlakeCache is a **multi-tenant** commercial Nix binary cache. The trust model —
who signs artifacts, who holds keys, how clients decide a path is genuine — is a
product-defining decision, so we grounded it in how the incumbents work:

- **cachix** — one Ed25519 keypair **per cache**; the **client generates and
  holds the secret** (zero-knowledge), signs narinfos before upload, and only the
  **public** key is registered and distributed into `trusted-public-keys`. The
  server never holds a tenant's signing secret. Private-cache *read* auth is a
  separate bearer token in `.netrc`.
- **attic** (our current `default` cache) — one keypair per cache, but the
  **server holds the secret** and signs on ingest.
- **FlakeHub Cache** — **no signing keypairs at all**; trust rides on
  **identity-aware JWT/OIDC** auth + netrc. No public-key distribution; the
  authenticated channel is the trust anchor.

Two orthogonal mechanisms fall out: **signature** (artifact integrity/authenticity,
portable, `require-sigs`) and **access auth** (who may read/write, token/JWT).

## Decision

1. **The trust boundary is per-cache**, never global and never per-node. Each
   cache (tenant cache) has one cryptographic identity; every node serving that
   cache uses that cache's key. A global key would break tenant isolation; a
   per-node key would force every builder to trust every node.

2. **Two composable layers, selectable per cache — the customer decides.** A
   cache's config picks its posture; we do **not** ship all modes at once, and a
   cache need not use every layer:
   - **Signature layer (integrity), pick one:**
     - **Zero-knowledge (default):** the customer holds the secret and signs
       before push; the cache stores the pre-signed narinfo and **verifies** on
       ingest against the cache's registered public key. FlakeCache holds **no**
       tenant signing secret — a fabric breach cannot forge any tenant's cache.
     - **Managed server-side signing (opt-in):** FlakeCache holds that cache's
       secret and signs on ingest (attic-style). Convenient; larger blast radius.
   - **Access layer (auth), independent:** public, or **identity-aware JWT/OIDC**
     (FlakeHub-style) with tokens in netrc. This fits CentralCloud's existing
     identity stack (HostBill → LLDAP, OIDC) and is a genuine differentiator, not
     a me-too. Read-auth is orthogonal to signatures: a cache may be signed,
     authed, or both.

3. **Incremental delivery.** Ship the signature layer first (it is the portable,
   any-Nix-client baseline); add the JWT/identity access mode later. The per-cache
   config gates which is active.

## The live `default` cache

`default` is our own single tenant; its key is `default:ESyvaQTi…`, already in
every builder's `trusted-public-keys`. It maps to **managed server-side signing**
with the **existing** key — the transparent drop-in that fixes the current
"narinfos served unsigned → CI can't substitute → rebuilds" break with **zero
fleet-wide nix.conf changes**. The `default` secret lives in **attic-postgres**
(the `attic-secrets` k8s secret holds only the HS256 token secret); it moves into
a k8s secret via ESO/OpenBao, RBAC-scoped.

## Consequences

- The signing/verifying primitive must be Nix-exact. It is: `crates/proto/nix`
  reproduces a real `nix store sign` signature byte-for-byte (known-answer test)
  and verifies a real `cache.nixos.org` signature (a path with a reference),
  pinning both the fingerprint format and the basename→full-path reference join.
- Zero-knowledge is the default posture and the stronger security story.
- Server-side signing is an explicit opt-in tier, holding only the keys of caches
  that chose it.
- `require-sigs` handling for auth-only caches (FlakeHub-style) is deferred to the
  access-layer work; not needed for the signature-layer baseline.

## Status

Built + verified: Ed25519 narinfo **signing** (server-side/`default` path) and
**parse + verify** (zero-knowledge ingest), both known-answer-tested against real
Nix. Pending: HTTP `.narinfo`/`nar` route wiring + per-cache key loading
(signature layer), then the JWT/identity access mode.
