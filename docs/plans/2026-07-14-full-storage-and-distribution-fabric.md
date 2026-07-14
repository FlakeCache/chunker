# FlakeCache full storage and distribution fabric

Status: research synthesis in progress

## Purpose contract

- **Purpose:** turn FlakeCache into a multi-tenant storage and distribution
  fabric that serves build caches, registries, and general S3-compatible object
  storage without coupling logical objects to a physical backend or region.
- **Consumers:** build clients, CI runners, package managers, registries,
  artifact producers, external S3 clients, operators, and billing/support.
- **Current problem:** protocol implementations and storage paths are split
  across the Elixir server, Rust fabric node, runner, builder, actions-cache,
  and backend-specific code. WebDAV works for sccache, but there is no
  multi-tenant S3 frontend or policy-driven placement authority.
- **Contract:** clients address stable tenant-scoped logical objects; the
  control plane authorizes and accounts for them; the fabric chunks, places,
  verifies, serves, repairs, rebalances, retains, and eventually deletes their
  bytes according to declared service objectives.
- **Evidence:** protocol conformance, compatibility matrices, migration tests,
  fault-injection tests, live canaries, placement audits, integrity scans,
  recovery drills, dedup/cost measurements, and clean repository gates.
- **Falsifier:** any supported client requires knowledge of physical storage;
  a declared failure-domain objective is not met; a tenant can infer or access
  another tenant's object; acknowledged data cannot be recovered; immutable
  data can be mutated or deleted before policy expiry; or migration loses an
  existing FlakeCache capability.

## Non-negotiable boundaries

1. `flakecache/server` remains the product and control plane during migration:
   tenancy, organizations, auth, billing, signing, registry metadata, policy,
   and current protocol compatibility remain live and reversible.
2. `flakecache/chunker` becomes the storage data plane: chunked CAS, nodes,
   placement, backend adapters, peer transfer, repair, rebalance, integrity,
   and byte-serving frontends.
3. CNPG is the authoritative consensus-backed metadata system. The data plane
   does not introduce Raft.
4. Garage is one S3-compatible backend. Local filesystems, mounted iSCSI/PVCs,
   remote S3 providers, Storage Box, and peer nodes are storage resources, not
   client-visible identities.
5. Logical bucket, cache, registry, package, and object identities remain stable
   while physical placement changes.
6. Production changes are delivered through Flux/GitOps after reversible
   canaries; direct cluster changes are diagnostic or explicitly temporary.

## Target model

```text
Nix | WebDAV | S3 | OCI | GHA | REAPI | package/registry protocols
                              |
                    protocol frontends
                              |
          logical namespace + auth + policy + accounting
                              |
        object/action metadata -------- immutable chunked CAS
                              |                   |
                         placement compiler + location index
                              |
         node disk | mounted block/file | Garage | remote S3 | archive
                              |
          custodians: verify | repair | rebalance | tier | GC
```

## Research inputs

This plan will incorporate source-indexed findings from:

- current FlakeCache repositories and production manifests;
- Google GFS/Colossus placement, encoding, curators, and custodians;
- S3 API and SigV4 compatibility requirements;
- Dynamo/Cassandra-style membership, anti-entropy, and failure handling;
- Ceph CRUSH and erasure-coding trade-offs;
- Garage placement and S3 behavior;
- Bazel Remote Execution CAS, Action Cache, and Remote Asset APIs;
- OCI Distribution and language package registries;
- sccache, Gradle, BuildKit, Buck2, and CI cache protocols;
- immutable retention, object lock, deletion, and multi-region DR systems.

### Adopted external lessons

| Source pattern | Adopt | Do not copy blindly |
|---|---|---|
| Google Colossus | logical storage objectives; separate metadata and bytes; direct node data path; background custodians; hot/cold media placement | Google-only client libraries or assumptions about one administrative trust domain |
| GFS | immutable/append-oriented large-object handling and continuous failure recovery | the original single-master metadata bottleneck |
| Dynamo | deterministic partitioning, failure-aware replica selection, hinted repair, and Merkle-style anti-entropy for immutable chunks | application conflict resolution for auth, billing, bucket policy, tags, or lifecycle state |
| Ceph CRUSH | weighted topology-aware placement across host/zone/region failure domains | exposing placement groups or backend topology in the product API |
| Garage | S3-compatible durable backend and region-aware replication | treating Garage buckets as FlakeCache logical buckets or Garage policy as the entire fabric policy |
| S3 | stable bucket/object API, SigV4, multipart/range/version/lifecycle/Object Lock semantics | binding client identities to a particular physical provider |
| Bazel REAPI | distinct Action Cache and immutable CAS; digest-first uploads; missing-blob discovery | forcing every build ecosystem through REAPI when a native protocol is simpler |
| OCI Distribution | mutable tags/repository scope separate from digest-addressed manifests and blobs; standard conformance suite | an OCI-only artifact model for package ecosystems with stronger native metadata contracts |

Sources: [Colossus architecture](https://cloud.google.com/blog/products/storage-data-transfer/a-peek-behind-colossus-googles-file-system),
[Colossus placement](https://cloud.google.com/blog/products/storage-data-transfer/how-colossus-optimizes-data-placement-for-performance/),
[Google Cloud durability](https://cloud.google.com/blog/products/storage-data-transfer/understanding-cloud-storage-11-9s-durability-target),
[GFS](https://research.google/pubs/the-google-file-system/),
[Dynamo](https://www.amazon.science/publications/dynamo-amazons-highly-available-key-value-store),
[Ceph CRUSH maps](https://docs.ceph.com/en/latest/rados/operations/crush-map-edits/),
[Bazel Remote APIs](https://github.com/bazelbuild/remote-apis), and
[OCI Distribution](https://github.com/opencontainers/distribution-spec/blob/main/spec.md).

### EARS and ACE boundary

EARS is a constrained natural-language authoring method, not a deployment or
runtime configuration format. The original method reduces ambiguity by using
ubiquitous, state-driven, event-driven, optional-feature, unwanted-behavior,
and combined patterns. It does not by itself prove completeness, correctness,
performance thresholds, safety, or executable test coverage.

Local source tracing found:

- `ace-coder` can classify patterns, suggest improvements, and generate test
  skeletons, but its generator may invent thresholds and uses process-dependent
  Python `hash()` values for test IDs.
- Singularity Forge ports the six patterns and returns a typed `SpecRecord`, but
  current validation is still a heuristic based on `SHALL`, length, and a small
  verb list. It does not prove domain semantics or execute generated tests.
- Engine's WorkSpec contract already supplies the missing execution boundary:
  stable requirements, acceptance criteria, scoped files/resources, test links,
  verification, rollback, run control, dependencies, and traceability.

Therefore ACE is an authoring/review producer, never the swarm's authority.
No generated criterion, threshold, test, topology, or policy becomes active
without source provenance, deterministic IDs, schema validation, explicit
approval, and executable evidence.

Sources: Mavin et al., [EARS, IEEE RE 2009](https://research.manchester.ac.uk/en/publications/easy-approach-to-requirements-syntax-ears/)
and [Ten Years of EARS](https://www.researchgate.net/publication/335535918_Ten_Years_of_EARS).

## Complete specification and execution package

The package passed through ACE/Purpose/Engine has distinct canonical layers:

1. **Purpose contract:** consumer, problem, outcome, non-goals, evidence,
   falsifier, doubt, and authority.
2. **Architecture and ADRs:** stable ownership and significant choices with
   rejected alternatives.
3. **Normative EARS specification:** one stable requirement ID per independently
   verifiable obligation, including unwanted behavior and optional features.
4. **Schema bundle:** logical object, manifest, catalog, service objective,
   placement decision, repair action, audit event, build action, and evidence.
5. **Traceability matrix:** requirement → acceptance criterion → test/eval →
   implementation owner/path → evidence → release gate.
6. **WorkSpec:** bounded scope, protected paths/resources, dependencies,
   red-first tests, verification commands, rollback, budgets, and run control.
7. **Conformance bundle:** golden protocol traces, SDK/client matrices,
   property/fault tests, fixtures, and expected typed failures.
8. **Deployment package:** versioned topology/policy configuration, secret
   references, migrations, manifests, rollout stages, and rollback revision.
9. **Evidence bundle:** immutable test, review, runtime, deployment, recovery,
   performance, security, and falsifier results linked to exact revisions.
10. **Approval envelope:** package digest, approver identity, policy decision,
    source revision, validity window, and signature.

Only a compiled runtime projection reaches the swarm:

```text
ACE research/draft
  -> approved EARS spec + schemas + WorkSpec
  -> deterministic compiler and conformance gates
  -> signed fabric-runtime/v1 projection
       catalog revision
       named service objectives
       protocol capability flags
       credential/policy references
       repair and admission budgets
       rollout revision
  -> swarm
```

The swarm verifies schema version, package digest, approval signature, catalog
revision, and objective satisfiability. It does not parse Markdown, infer
requirements, invent defaults, or execute arbitrary agent-generated commands.

### Typed handover gates

The useful lesson from Spec Kit, OpenSpec, Kiro, and FRET is not another set of
Markdown templates. It is that lifecycle transitions are first-class state
changes. Spec Kit persists workflow state and pauses at review gates; OpenSpec
keeps proposed deltas separate from current truth until archive; Kiro separates
requirements, design, tasks, and execution; FRET adds a semantic-analysis
boundary between structured language and formal or executable verification.

FlakeCache therefore treats every handover as a content-addressed envelope,
not an agent passing prose or chat context to another agent. Every envelope has
a stable change ID, input and output artifact digests, schema versions, producer
and accepting authority, required validators, evidence references, decision,
timestamp, and reject/resume reason. A consumer shall reject missing, stale,
unsigned, schema-invalid, or unauthorized envelopes.

The Need gate itself has an upstream provenance chain:

```text
vision and enduring principles
  -> strategic themes and constraints
  -> portfolio outcomes and measurable objectives
  -> stakeholder expectations + operational evidence + opportunities
  -> problem framing and alternatives
  -> accepted need
  -> purpose contract
```

Vision states the desired future and boundaries; it does not authorize a
feature. Strategy chooses where to act. Portfolio objectives make the intended
change measurable. A need identifies one solution-neutral problem that blocks
an objective, with affected stakeholders, baseline evidence, urgency, value,
constraints, assumptions, and the consequence of doing nothing. The Need gate
rejects work that cannot trace upward, duplicates an active need, prescribes a
solution prematurely, lacks decision authority, or has no observable baseline.
Conversely, a higher-level objective with no accepted child needs is visible as
an execution gap rather than being silently treated as covered.

Traceability is bidirectional: every requirement traces through purpose and
need to an objective and vision revision, while runtime outcomes, incidents,
cost, and falsifier evidence flow upward. Evidence may revise a need, objective,
strategy, or even the vision; it never silently rewrites an accepted artifact.
This follows NASA's needs-goals-objectives model and its requirement that lower
levels trace to mission or system scope while stakeholder involvement forms a
self-correcting feedback loop.

| Gate | Accepted input | Required output | Accepting authority and falsifier |
|---|---|---|---|
| Need | versioned vision/objective links, stakeholder expectations, research, and operational evidence | accepted solution-neutral need and purpose contract | portfolio/product authority; reject if no upward trace, named consumer, baseline, outcome, or falsifier |
| Specify | approved purpose | requirement set, glossary, scenarios, non-goals | domain owner; reject ambiguity, unverifiable obligations, invented thresholds, or uncovered unwanted behavior |
| Formalize | EARS requirements and domain model | typed schemas, invariants, temporal/property assertions | spec compiler/subject expert; reject semantic loss or an unmapped requirement |
| Design | approved requirements and schemas | ADRs, threat model, ownership and failure model | architecture/security owners; reject an unexplained trade-off or authority conflict |
| Decompose | approved design | WorkSpec task graph with paths, dependencies, budgets, rollback, and tests | execution planner; reject unbounded tasks or missing requirement/test links |
| Dispatch | ready WorkSpec | signed capability-scoped work leases | scheduler/operator policy; reject overlap, missing capability, stale base revision, or absent resource budget |
| Integrate | implementation revisions and task evidence | reviewed candidate revision and traceability update | code owners and CI policy; reject failing tests, unresolved review, scope drift, or lost provenance |
| Release | candidate, deployment package, and evidence bundle | signed release/promotion decision | release/security policy; reject failed conformance, migration, rollback, security, or SLO gates |
| Operate | released projection and observed telemetry | runtime evidence, incidents, repair actions, and new change proposals | control plane/operator; reject unauthorized drift and feed falsifiers back to the Need or Specify gate |

Gate state is `draft -> ready -> accepted | rejected -> superseded`; rejection
is resumable from the same persisted gate after new evidence, while superseding
creates a new immutable envelope linked to its predecessor. Fan-out creates
child WorkSpecs with disjoint leases; fan-in requires every required child
digest or an explicit approved waiver. Humans or policy engines approve risk;
agents may propose, validate, and execute but cannot self-approve a gate they
produced. Emergency operation uses the same envelope with a break-glass policy,
short validity window, named authority, and mandatory retrospective evidence.

This combines [Spec Kit's persisted workflows and review gates](https://github.github.com/spec-kit/reference/workflows.html),
[OpenSpec's proposal/apply/archive separation](https://github.com/Fission-AI/OpenSpec),
[Kiro's requirements/design/tasks workflow](https://kiro.dev/docs/cli/v3/specs/),
and [NASA FRET's structured-requirement analysis](https://github.com/NASA-SW-VnV/fret),
with [NASA's stakeholder-expectations and needs-goals-objectives flow](https://www.nasa.gov/reference/4-1-stakeholder-expectations-definition/)
without making any one authoring tool the runtime source of truth.

## Protocol and distribution portfolio

One protocol adapter must never become another storage engine. Every frontend
maps mutable protocol metadata to the control plane and immutable bytes to the
same fabric CAS.

| Ecosystem | Preferred frontend | Immutable data | Mutable metadata | Initial priority |
|---|---|---|---|---|
| sccache / Rust / C / C++ | S3, retain WebDAV during migration | compiler entries | cache namespace and expiry | P0 |
| Nix | Nix binary-cache HTTP | NARs and signed narinfo payloads | cache policy and roots | P0 preserve |
| Forgejo/GHA | Actions cache API | committed archives | reservations, keys, scopes | P0 preserve |
| OCI / BuildKit | OCI Distribution | layers, configs, digest manifests | tags, repositories, uploads | P1 |
| Bazel / Buck2 / compatible builders | REAPI CAS + Action Cache + ByteStream | blobs and trees | action results, leases | P1 |
| Gradle | Gradle HTTP remote cache | build-cache entries | keys and retention | P1 |
| Cargo | sparse registry + crate download; S3 only for compiler cache | crate archives and index snapshots | versions, yanks, index heads | P2 |
| Hex / Gleam | native package registry/download | package tarballs and docs | releases, retirement, metadata | P2 |
| Maven | Maven repository HTTP | artifacts and checksums | versions and repository policy | P2 |
| npm / pnpm / Yarn | npm registry HTTP | tarballs | package/version metadata and dist-tags | P2 |
| PyPI / pip / uv | Simple Repository API + file endpoints | wheels and sdists | project/version metadata | P2 |
| Go | module proxy protocol | module zip, mod, info | version lists and retractions | P2 |
| Turbo / Nx | native remote-cache HTTP adapters | task artifacts | task keys and signatures | P2 |
| ccache | HTTP/S3 adapter after client compatibility proof | compiler entries | namespace and expiry | P2 |
| Git LFS | LFS batch/transfer API | LFS objects | repository references and locks | P3 |
| models and datasets | OCI artifacts first; S3 for general access | model/data shards | aliases, cards, lineage | P3 |
| general external storage | S3-compatible API | object versions and multipart parts | buckets, ACL/policy, lifecycle, locks | staged P1-P4 |

Adapters are accepted only with an identified live consumer and conformance or
golden-client proof. A generic GHA archive can cache language directories, but
that does not replace a native package registry or build-cache protocol.

## Build execution and fan-out

Storage and execution are one product fabric but separate failure and security
boundaries:

```text
Git/Actions/CLI
      |
build gateway -- identity, policy, graph, quota, admission
      |
scheduler -- capabilities, locality, fairness, cancellation, retries
      |
      +-- sccache-dist workers
      +-- Nix remote builders
      +-- Bazel/Buck2 REAPI workers
      +-- BuildKit workers
      +-- Moon/Turbo/Nx task workers
      +-- isolated generic Actions runners
                 |
          shared CAS and logs
```

- `flakecache/builder` owns scheduling and supported remote-build adapters.
- `flakecache/runner` owns isolated execution, worker lifecycle, logs, and
  cancellation. It does not implement another artifact store.
- `flakecache/server` owns product identity, project policy, quotas, billing,
  provenance, and user-facing job state.
- `flakecache/chunker` owns CAS transfer, input materialization, output capture,
  locality information, and immutable evidence blobs.
- Large fan-out sends action digests and small descriptors. Workers fetch
  missing inputs from the nearest CAS location and publish outputs once by
  digest; source archives are not copied through the scheduler.
- Scheduling considers platform/toolchain capability, tenant fairness, data
  locality, queue age, failure history, cost, and egress. Cache locality is a
  preference, never authorization.
- Speculative or duplicate execution may race, but immutable output digest
  publication is idempotent and billing records every attempt separately.
- Nix, REAPI, sccache-dist, and task-graph adapters keep their native client
  contracts. The gateway normalizes their internal action and evidence model;
  it does not pretend the protocols are identical.

Execution portfolio: native Nix remote builds, sccache-dist for Rust/C/C++,
Bazel/Buck2 REAPI, BuildKit, Moon, Turbo, Nx, Gradle workers, and generic
Actions jobs. Each adapter requires a live consumer, hermeticity statement,
toolchain identity, cancellation proof, cache-poisoning defense, and fan-out
load test before production promotion.

## Publication, promotion, and supply chain

Registries and Actions publish through one promotion graph rather than copying
opaque archives between unrelated stores:

```text
ephemeral build output
  -> verified artifact
  -> registry/package candidate
  -> signed release
  -> immutable retained release + provenance
```

Promotion attaches or verifies checksums, SBOMs, signatures, attestations,
source revision, builder identity, toolchain lock, test evidence, policy result,
and release channel. It changes roots, retention, placement objectives, and
visibility; existing immutable chunks are reused. Supported distribution
classes include package and OCI registries, CLI/installers, documentation,
firmware, WASM, VM/container images, Git LFS, and model/data artifacts.

## Service-objective policy

Users choose objectives; the placement compiler chooses copies/fragments.

| Class | Intended content | Regional objective | Deletion/retention |
|---|---|---|---|
| ephemeral | PR and reproducible compiler cache | one region; eviction allowed | TTL/LRU |
| standard | normal build cache and mirrors | survive declared node/zone failures | policy retention |
| premium | important private artifacts | survive one region loss; active regional reads | retained versions |
| immutable | releases, signatures, SBOM/provenance, audit | two regions plus independent archival failure domain | WORM/Object Lock and legal hold |

Backend-internal redundancy counts only when its failure domains and repair
contract are known. A hot node copy is a latency copy, not a durability copy.
The compiler must reject an unsatisfiable policy rather than silently weaken it.

## Metadata and consistency rules

- Bucket creation, identity, credentials, policy, quotas, version heads, tags,
  retention, legal holds, billing, and deletion authorization are strongly
  consistent control-plane state in CNPG.
- Chunk bytes are immutable and content-addressed. Placement and health may
  converge asynchronously because the desired state is deterministic and
  repairable.
- A write is acknowledged only after its selected service-class durability
  threshold is met and its manifest/root is committed. Local buffering alone
  cannot acknowledge a durable write.
- Read-after-write and list semantics are defined at logical-object level;
  backend eventual consistency must not leak through the frontend.
- Deletion creates a durable tombstone first. Physical chunks are reclaimed
  only after version, retention, legal-hold, replica, and shared-reference
  checks pass.

## Security and tenant isolation

- Logical buckets use opaque immutable IDs plus optional validated display
  names. Physical backend buckets, keys, paths, regions, and topology remain
  private.
- SigV4 credentials are scoped to tenant, logical bucket, actions, prefixes,
  expiry, and optional network/workload identity; secrets are hashed or
  envelope-encrypted and rotatable.
- Cross-tenant physical dedup is disabled initially. Organization-scoped dedup
  is the safe default until encryption, deletion, billing, and timing-leak
  contracts prove a broader trust boundary.
- Every placement, policy, credential, retention, repair, and deletion change
  emits an immutable audit event with actor, reason, old/new state, and trace ID.
- Object checksums are verified on ingress, after transfer, during scrubs, and
  before repair promotion. Corrupt replicas are quarantined, never voted good.

## Capability disposition

The final matrix must classify each discovered capability as **preserve**,
**consolidate**, **extend**, **replace**, or **retire**, with its source path,
canonical owner, migration proof, and rollback path. It must cover at least:

- Nix binary cache and zero-knowledge signing;
- sccache WebDAV and distributed compilation;
- S3 frontend and general object storage;
- GHA/Forgejo Actions cache;
- OCI registry and artifact distribution;
- REAPI/CAS/Action Cache/Remote Asset;
- Cargo/crates, Hex/Gleam, npm/pnpm/Yarn, PyPI/uv, Maven/Gradle, and Go;
- Turbo/Nx/Buck2/BuildKit/ccache;
- CLI downloads, Git LFS, model/data artifacts, and generic immutable blobs;
- local, filesystem/mounted, S3, Storage Box, peer, tiered, and archival storage;
- tenancy, auth, quotas, billing, lifecycle, immutability, audit, and telemetry.

## Delivery phases

### Phase 0 — contracts and conformance harness

Freeze ownership, object/manifest/placement schemas, consistency semantics,
service classes, threat model, protocol matrices, migration invariants, and
failure-injection scenarios. Build protocol clients and golden traces before
changing production routing.

### Phase 1 — multi-tenant S3 build-cache vertical slice

Add opaque logical bucket allocation, scoped access credentials, SigV4
verification, and the S3 operations exercised by supported sccache versions.
Store objects through the existing chunked CAS and configured warm/cold backend.
Generate client configuration and prove miss, put, hit, overwrite semantics,
delete, credential isolation, restart recovery, and Garage outage behavior.

### Phase 2 — placement catalog and policy compiler

Represent regions, zones, nodes, backend instances, capabilities, costs,
capacity, health, trust, and failure domains. Compile service objectives into
placements without exposing topology to clients. Begin with replicated chunks;
introduce erasure coding only after measured object-size and repair economics.

### Phase 3 — custodians and regional operation

Add integrity scrubbing, anti-entropy, under-replication repair, rebalancing,
read promotion, cold demotion, capacity evacuation, tombstones, safe GC,
bounded repair bandwidth, and operator-visible explanations for every move.

### Phase 4 — external S3 compatibility

Add multipart upload, ranges, conditional requests, listing/pagination,
presigned URLs, checksums/ETags, versioning, lifecycle, notifications, bucket
policies, object lock, legal holds, replication, metering, and abuse controls.
Certify against AWS SDKs and CLIs across multiple languages.

### Phase 5 — build-cache and registry distribution portfolio

Route existing Nix, WebDAV, GHA, OCI, and CLI flows through the fabric, then add
REAPI and prioritized package/registry adapters. Keep mutable names, tags,
versions, and action results separate from immutable content blobs. Add origin
shielding, pull-through mirrors, regional delivery, and artifact promotion.

### Phase 6 — migration and production proof

Introduce `CasBackend` per-cache canaries in the Elixir server, dual-read or
shadow verification where safe, rollback to existing S3 paths, GitOps rollout,
region/backend failure drills, recovery-time evidence, cost/dedup/latency
measurements, and gradual default cutover.

## Completion gate

Completion requires requirement-by-requirement evidence for protocol
compatibility, tenant isolation, durability objectives, regional failover,
integrity/repair, lifecycle/immutability, quotas/metering, observability,
migration/rollback, Nix/Just/devenv gates, committed and pushed repositories,
and removal of every integrated session workspace. Passing one protocol smoke
or one crate test cannot prove this plan complete.
