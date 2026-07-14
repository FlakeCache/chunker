# ADR 0001: FlakeCache Distributed Storage Fabric

- Status: Proposed
- Date: 2026-07-14

## Context

FlakeCache began as a Nix binary cache with content-defined chunking. The same
stored bytes increasingly need to serve Nix, OCI, npm, S3, and file or block
consumers without building an independent storage stack for every protocol.
Duplicating objects by protocol would discard the principal advantage of the
existing FastCDC chunker: deduplication across artifacts, formats, tenants, and
locations.

A distributed design also changes which state may remain centralized. Chunk
ingest, lookup, placement, repair, and reference tracking are on the byte path
and must continue through node or region failures. Tenant administration,
billing, tag authority, and global coordination are authoritative but
comparatively infrequent. Treating both classes of state alike would make a
central database the availability and scaling boundary for every object
operation.

The storage fabric therefore needs one durable content model, multiple protocol
adapters, modular location backends, and a control plane that can evolve without
remaining in the byte path.

## Decision

### One content-addressed storage fabric

FlakeCache will evolve into a distributed, content-addressed, self-healing
storage fabric. S3, OCI, npm, Nix binary cache, and WebDAV are protocol views
over one deduplicated chunk store. NFS will follow those immutable and object
protocols. iSCSI will be last.

Each storage location has a modular backend. Initial backend types include
GlusterFS, S3-compatible storage, and a Hetzner Storage Box cold tier. A backend
is a placement and durability target; it does not define a separate artifact
model or deduplication domain.

Cache nodes are the storage servers and own the data. The swarm keeps hot
replicas on its nodes and writes through to a cold tier for durability. This is
not a cache placed in front of Garage. Garage may be configured as one S3
backend, with the same status as other backend implementations.

### Leaderless data path

Raft will not be placed on the data path. Content-addressed chunk writes are
idempotent: equal hashes identify equal bytes, while unequal bytes cannot
legitimately contend for the same identity. The swarm will coordinate with:

- SWIM membership and failure detection;
- rendezvous hashing for stable, consistent placement;
- Merkle-tree anti-entropy for discovery and repair of divergent replicas; and
- CRDT state where coordination cannot be reduced to immutable content.

This follows Garage's leaderless Dynamo-and-CRDT model. Consensus remains
available to the low-frequency control plane where a single authoritative
decision is required; it is not required to acknowledge ordinary chunk writes
or reads.

### Embedded node metadata and distributed garbage collection

Every node will keep its hot-path metadata in embedded on-disk stores. The
chunk -> manifest -> artifact -> refcount DAG will be stored in `redb`, using
its crash-safe, memory-mapped, disk-resident storage. Reachability and garbage
collection will run graph algorithms over that DAG rather than loading a global
index into memory.

The existing scaffolded bespoke graph WAL is not the storage engine. It is
non-atomic and unfinished. FlakeCache will use `redb` now and revisit a custom
memory-mapped page store only if measured traversal performance becomes a
bottleneck.

Near-duplicate and similarity search is a phase-two concern. It will use an
on-disk vector index based on DiskANN with RaBitQ quantization; it will not add a
central vector service to the hot path.

Distributed refcounts will be represented as CRDT PN-counters. Collection is a
conservative reachability sweep with grace periods and anti-entropy convergence,
not an immediate delete triggered by one node's local count.

### Central authority outside the byte path

Central PostgreSQL, operated by CloudNativePG and potentially out of region,
will hold only low-frequency authoritative state:

- tenants and entitlement policy;
- billing state;
- mutable tag authority; and
- global garbage-collection coordination.

It may be proxied when a control-plane operation requires it. It will not be
queried for each chunk, manifest, artifact read, or artifact write. The central
authority will issue signed, scoped tokens; swarm nodes verify those tokens
locally on requests.

### Stable chunk identity

Content-defined chunking is provided by the existing in-house FastCDC chunker
crate. The store format is permanently committed to FastCDC v2020 for this store
version. A content-addressed deduplication store must not change its cut points
without an explicit store-version migration and coexistence plan.

FastCDC v2020 is the crate-recommended implementation, produces the same cut
points as v2016, and is faster. The `ronomon` fork produces different cut points
and is therefore not a compatible implementation or runtime substitute.

### Durable on-disk format

The on-disk format will reuse proven design elements from ruvector's `rvf`:

- a two-level manifest;
- a `REFCOUNT` segment;
- append-only crash safety;
- a SHAKE-256 witness chain;
- copy-on-write branching; and
- Ed25519 and ML-DSA signing.

Those structures will be re-keyed to content hashes. `rvf-crypto` will be lifted
as a standalone crate. The vector-cluster-keyed `rvf` runtime will not be
ported.

VectorDrive is canonical for graph and vector capabilities. Ruvector is a
frozen, read-only donor from the same `rUv` lineage; VectorDrive is the
persistent, documented, productized superset. FlakeCache may harvest the
sparsifier, possibly SPANN after evaluation, and the `rvf` format and crypto
design. The same algorithm will never be maintained in both trees.

### Protocol staging and block semantics

Implementation order is:

1. Immutable and object protocols: S3, OCI, npm, and Nix binary cache, with
   WebDAV where it preserves the same object semantics.
2. File semantics through NFS.
3. Block semantics through iSCSI.

iSCSI will be built only behind a low-latency write-back journal, referred to as
LLBS, over the immutable content-addressed store. The journal absorbs ordered,
mutable block writes and publishes immutable snapshots or extents into the CAS.
The CAS itself will not pretend to provide in-place block mutation.

### Target Rust layout and Elixir control plane

The target Rust workspace is:

```text
core/
  chunker/
  crypto/
  cas/
  meta/
swarm/
  membership/
  placement/
  antientropy/
  router/
backend/
  glusterfs/
  s3/
  hetzner-storage-box/
proto/
  s3/
  oci/
  npm/
  nix/
  webdav/
  nfs/
  journal/  # last, with iSCSI
  iscsi/    # last
node/
token/
```

The existing Elixir server remains the control plane during the transition. It
continues to own tenant-facing orchestration and authoritative workflows while
Rust nodes progressively take protocol serving, chunk IO, placement, repair,
and local token verification. The byte path will be shed incrementally, with
observable compatibility gates, rather than replaced in one cutover.

## Consequences

### Benefits

- One chunk identity deduplicates content across every protocol and backend.
- Idempotent writes and leaderless repair remove a consensus round trip from
  normal data operations.
- Hot replicas remain available within the swarm while cold-tier write-through
  supplies a separate durability boundary.
- Embedded disk-resident metadata scales with storage nodes and avoids a
  central per-object metadata bottleneck.
- Protocols, backends, and locations can evolve independently around the same
  CAS contract.
- The Elixir-to-Rust transition can be staged without prematurely rewriting the
  control plane.

### Costs and hard correctness problems

- **iSCSI over immutable chunks is intrinsically difficult.** Ordered writes,
  flush and durability barriers, read-after-write behavior, partial-block
  updates, snapshots, and crash recovery belong in LLBS. iSCSI cannot ship
  until the journal proves those semantics under failure.
- **Global garbage collection is distributed-systems work.** Per-node metadata
  means no local refcount is globally authoritative. PN-counters, Merkle
  anti-entropy, conservative marking, grace periods, tombstones, and cold-tier
  retention must prevent premature deletion during partitions and delayed
  replication.
- **Multi-writer consistency remains hard without Raft.** Immutable chunks are
  conflict-free, but mutable names, tags, manifests under construction,
  refcount deltas, membership changes, and concurrent repair are not
  automatically so. Their merge rules, fencing, idempotency keys, and authority
  boundaries must be explicit and tested under partitions.
- Write-through durability couples ingest acknowledgement policy to cold-tier
  health. The system must define when degraded writes are rejected, queued, or
  acknowledged and expose that state operationally.
- Store versioning becomes permanent infrastructure. Chunking parameters,
  hashing, manifests, signatures, and witness-chain rules require explicit
  migration paths rather than in-place reinterpretation.
- Local token verification reduces central latency but requires bounded token
  lifetime, key rotation, revocation strategy, tenant scoping, and clock-skew
  handling.

## Alternatives

### JuiceFS

Rejected as the foundation because its external metadata service recreates the
central-metadata dependency this design removes from the hot path. It can expose
useful filesystem semantics, but those semantics would sit above a metadata
architecture that does not match FlakeCache's leaderless, node-owned CAS.

### Ceph

Rejected because its operational and resource weight is disproportionate to
the product, and large CRUSH placement changes can cause substantial rebalancing
IO. FlakeCache needs application-aware content identity, cross-protocol
deduplication, and modular cold backends rather than a second general-purpose
distributed storage platform beneath them.

### Central PostgreSQL for hot-path metadata

Rejected as the scaling and availability wall. A global database lookup or
mutation for each chunk would couple byte throughput and regional availability
to database latency, connection capacity, and failover.

The analogous ZFS lesson is that deduplication metadata is touched for every
deduplicated block write or free. OpenZFS documents that when the DDT cannot be
kept cached, misses become random disk reads and performance collapses; it
estimates slightly more than 320 bytes of memory per cached DDT entry. FlakeCache
will keep its metadata disk-resident and locality-aware rather than make a
central, effectively RAM-pressure-sensitive index the prerequisite for every
operation. See [OpenZFS Workload Tuning: Deduplication](https://openzfs.github.io/openzfs-docs/Performance%20and%20Tuning/Workload%20Tuning.html#deduplication).

### Garage as the storage owner

Rejected because it would make FlakeCache a protocol cache in front of another
distributed system and surrender ownership of placement, repair, hot replicas,
and the cross-protocol content DAG. Garage's coordination model is an input to
the design, and Garage remains usable as an S3 backend, but FlakeCache nodes own
the fabric.
