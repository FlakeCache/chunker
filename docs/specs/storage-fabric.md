# FlakeCache Storage Fabric Contract

> **Status:** Planned normative contract; implementation partial
> **Owner:** FlakeCache storage data plane and product control plane
> **Consumers:** Protocol frontends, registries, build schedulers, runners,
> operators, billing, and external object-storage clients
> **Last verified:** 2026-07-14

## Purpose

FlakeCache provides tenant-scoped logical objects, build caches, registry
distribution, and S3-compatible storage over a backend-independent,
content-addressed, policy-placed storage fabric. Clients address stable logical
names while FlakeCache selects physical nodes, regions, media, and providers.

The governing architecture and implementation sequence are defined by
[the full storage and distribution plan](../plans/2026-07-14-full-storage-and-distribution-fabric.md).
The disk-first catalog is defined by
[the v1 JSON Schema](schemas/fabric-config-v1.schema.json); the deployment tree
contains a non-authoritative example.

## Current state

The production Elixir server provides Nix, sccache WebDAV, GHA cache, registry,
tenancy, auth, and billing surfaces over CNPG and Garage. The Rust fabric has a
chunked CAS, filesystem and S3 backends, warm/cold tiering, deterministic node
placement, Nix/WebDAV serving, token verification, and a versioned YAML
resource/service-class loader. Multi-tenant S3 serving, policy execution,
regional repair, lifecycle, external S3 compatibility, and remote-execution
integration are not yet complete.

## Requirements convention

Normative requirements use stable IDs and EARS patterns:

- ubiquitous: `The FlakeCache system shall ...`;
- event-driven: `When ..., the FlakeCache system shall ...`;
- state-driven: `While ..., the FlakeCache system shall ...`;
- optional feature: `Where ..., the FlakeCache system shall ...`; and
- unwanted behavior: `If ..., then the FlakeCache system shall ...`.

Each requirement states one independently verifiable obligation. Text without
a requirement ID is informative.

## Authority boundaries

| Concern | Authority | Fabric role |
|---|---|---|
| Tenant, organization, credentials, policy, quota, billing | Elixir control plane and CNPG | Enforce signed/scoped decisions |
| Logical bucket/object/version/tag/action metadata | Control plane and CNPG | Resolve immutable manifests |
| Chunk bytes and physical placement | Rust storage fabric | Canonical data plane |
| Registry names, releases, tags, yanks, channels | Owning registry/control plane | Store referenced immutable content |
| Build scheduling, queues, retries, cancellation | Builder scheduler | Consume CAS locality/capability signals |
| Runner isolation and execution | Runner/worker plane | Materialize inputs and publish outputs |
| Physical backend durability | Backend provider plus fabric catalog | Verify and compose declared guarantees |
| Deployment and production rollout | Flux/GitOps | Reconcile reviewed desired state |

## Terminology

| Term | Meaning |
|---|---|
| Logical bucket | Stable tenant-scoped S3 namespace unrelated to a physical provider bucket |
| Logical object | Bucket/key/version identity resolving to one immutable manifest |
| Manifest | Ordered object metadata and chunk references with integrity information |
| Chunk | Immutable content-addressed byte range |
| Resource | Named physical storage target with backend, region, zone, capability, and health |
| Service objective | Declarative durability, availability, locality, retention, and cost requirements |
| Placement plan | Deterministic resource selection satisfying one service objective |
| Custodian | Background verifier, repairer, rebalancer, tiering, or GC worker |
| Root | Live logical reference that prevents manifest/chunk collection |
| Tombstone | Durable deletion intent preceding physical reclamation |
| Promotion | Policy transition from ephemeral output to verified, released, or immutable artifact |

## Namespace and identity requirements

- **FC-NS-001 — Ubiquitous.** The FlakeCache system shall assign every tenant,
  logical bucket, logical object version, manifest, chunk, resource, placement,
  credential, build action, and audit event a stable immutable identifier.
- **FC-NS-002 — Ubiquitous.** The FlakeCache system shall keep logical bucket
  identity independent of physical backend bucket, path, node, and region.
- **FC-NS-003 — Event-driven.** When a client addresses an unknown tenant or
  bucket, the FlakeCache system shall return the protocol-defined not-found or
  access-denied response without revealing physical topology.
- **FC-NS-004 — Unwanted behavior.** If authenticated tenant scope differs from
  the requested namespace, then the FlakeCache system shall deny the operation
  before reading object, manifest, placement, or billing data.
- **FC-NS-005 — Ubiquitous.** The FlakeCache system shall preserve namespace,
  object, and version identifiers while placement or backend topology changes.

## Configuration and policy requirements

- **FC-POL-001 — Ubiquitous.** The FlakeCache system shall load regions,
  zones, resources, backend kinds, and named service objectives from a
  versioned declarative catalog rather than compiled constants.
- **FC-POL-002 — Event-driven.** When a catalog uses an unsupported schema
  version, the FlakeCache system shall reject it without applying partial state.
- **FC-POL-003 — Event-driven.** When resource identities, topology, or a
  service objective are malformed or unsatisfiable, the FlakeCache system shall
  reject the catalog or policy instead of silently weakening durability.
- **FC-POL-004 — State-driven.** While disk-first configuration is active, the
  FlakeCache system shall treat the validated YAML revision as the placement
  input and record its digest with each resulting decision.
- **FC-POL-005 — Optional feature.** Where CNPG-backed configuration is enabled,
  the FlakeCache system shall supply the same typed catalog records and preserve
  schema revision, provenance, and deterministic placement behavior.
- **FC-POL-006 — Ubiquitous.** The FlakeCache swarm shall accept only a typed,
  versioned runtime projection and shall not interpret EARS prose, Markdown,
  agent prompts, or arbitrary generated commands as runtime policy.
- **FC-POL-007 — Event-driven.** When a runtime projection is activated, the
  FlakeCache system shall verify its schema, digest, source revision, approval,
  catalog revision, and objective satisfiability before replacing active state.
- **FC-POL-008 — Unwanted behavior.** If an agent-generated requirement,
  threshold, test, or policy lacks provenance and explicit approval, then the
  FlakeCache system shall exclude it from the compiled runtime projection.

## Placement requirements

- **FC-PLC-001 — Ubiquitous.** The FlakeCache system shall derive desired
  placement deterministically from content identity, validated catalog, and
  selected service objective.
- **FC-PLC-002 — Ubiquitous.** The FlakeCache system shall distinguish resource,
  node, zone, region, provider, and archival failure domains.
- **FC-PLC-003 — Event-driven.** When a policy requires multiple failure
  domains, the FlakeCache system shall select distinct eligible domains before
  adding redundant copies within an already selected domain.
- **FC-PLC-004 — Ubiquitous.** The FlakeCache system shall count backend-internal
  redundancy only when the catalog declares its failure-domain and repair
  contract.
- **FC-PLC-005 — Ubiquitous.** The FlakeCache system shall classify latency
  copies separately from acknowledged durability copies.
- **FC-PLC-006 — Unwanted behavior.** If eligible resources cannot satisfy the
  objective at admission time, then the FlakeCache system shall reject or
  explicitly queue the write rather than acknowledge reduced durability.
- **FC-PLC-007 — Event-driven.** When catalog membership changes, the
  FlakeCache system shall compute affected placements without remapping
  unrelated content unnecessarily.

## Write and integrity requirements

- **FC-WRT-001 — Event-driven.** When object bytes are accepted, the
  FlakeCache system shall verify the client checksum when supplied and compute
  the canonical object and chunk digests.
- **FC-WRT-002 — Event-driven.** When a write is acknowledged, the FlakeCache
  system shall have committed its manifest/root and met the selected durability
  acknowledgement threshold.
- **FC-WRT-003 — Ubiquitous.** The FlakeCache system shall make repeated
  publication of identical chunks and manifests idempotent.
- **FC-WRT-004 — Unwanted behavior.** If any stored or transferred bytes fail
  digest verification, then the FlakeCache system shall quarantine that copy
  and exclude it from durability acknowledgement and repair sources.
- **FC-WRT-005 — Ubiquitous.** The FlakeCache system shall stream bounded
  uploads and downloads without buffering an unbounded complete object in BEAM
  or a protocol gateway.

## Read and consistency requirements

- **FC-RED-001 — Event-driven.** When a logical object read succeeds, the
  FlakeCache system shall return bytes matching the selected immutable version
  and its declared checksum.
- **FC-RED-002 — Event-driven.** When the nearest copy is absent or corrupt, the
  FlakeCache system shall try another authorized healthy placement and record
  degraded service evidence.
- **FC-RED-003 — Ubiquitous.** The FlakeCache system shall provide its declared
  logical read-after-write, listing, conditional, and version semantics even
  when a physical backend is eventually consistent.
- **FC-RED-004 — Event-driven.** When a remote read is promoted into a hot tier,
  the FlakeCache system shall verify content identity before exposing the copy.

## Repair and lifecycle requirements

- **FC-RPR-001 — Ubiquitous.** The FlakeCache system shall continuously detect
  missing, corrupt, misplaced, over-replicated, and under-replicated chunks.
- **FC-RPR-002 — Event-driven.** When desired and observed placement differ,
  the FlakeCache system shall create a bounded, idempotent repair or rebalance
  action with source, target, reason, and policy revision.
- **FC-RPR-003 — State-driven.** While a backend or region is degraded, the
  FlakeCache system shall rate-limit repair to preserve foreground service and
  avoid correlated retry storms.
- **FC-LIF-001 — Event-driven.** When deletion is authorized, the FlakeCache
  system shall persist a tombstone before removing roots or physical chunks.
- **FC-LIF-002 — Unwanted behavior.** If any live version, root, retention
  period, legal hold, shared reference, or repair obligation protects a chunk,
  then the FlakeCache system shall not reclaim it.
- **FC-LIF-003 — Optional feature.** Where immutable retention is enabled, the
  FlakeCache system shall prevent overwrite and premature deletion across every
  acknowledged failure domain.

## S3 requirements

- **FC-S3-001 — Ubiquitous.** The FlakeCache S3 frontend shall authenticate
  supported header-signed and presigned SigV4 requests using scoped, rotatable
  credentials.
- **FC-S3-002 — Event-driven.** When an S3 access key is accepted, the
  FlakeCache system shall authorize its tenant, logical bucket, action, prefix,
  expiry, and optional workload/network restrictions.
- **FC-S3-003 — Ubiquitous.** The FlakeCache system shall implement the exact
  S3 methods, status codes, XML errors, ETag/checksum behavior, ranges,
  conditions, pagination, and multipart semantics claimed by its compatibility
  profile.
- **FC-S3-004 — Event-driven.** When a build-cache profile is provisioned, the
  FlakeCache system shall generate working sccache endpoint, region, logical
  bucket, key-prefix, and scoped credential configuration.
- **FC-S3-005 — Unwanted behavior.** If signature canonicalization, payload
  digest, request time, credential scope, or authorization fails, then the
  FlakeCache system shall reject the request before object mutation.
- **FC-S3-006 — Optional feature.** Where general external storage is enabled,
  the FlakeCache system shall publish an explicit compatibility profile and
  pass its SDK/CLI conformance matrix before claiming S3 support.

## Build, runner, and registry requirements

- **FC-BLD-001 — Ubiquitous.** The Build Fabric shall separate scheduling and
  execution state from immutable CAS input and output bytes.
- **FC-BLD-002 — Event-driven.** When an action is dispatched, the scheduler
  shall send content identities and bounded descriptors rather than duplicate
  complete source or dependency archives.
- **FC-BLD-003 — Event-driven.** When a worker lacks an input, it shall fetch
  and verify that input from an authorized placement before execution.
- **FC-BLD-004 — Unwanted behavior.** If worker platform, toolchain, isolation,
  tenant, or authorization does not satisfy the action contract, then the Build
  Fabric shall not dispatch the action to that worker.
- **FC-BLD-005 — Optional feature.** Where fan-out or speculative execution is
  enabled, the Build Fabric shall keep output publication idempotent and record
  every attempt, cancellation, cost, and terminal status.
- **FC-REG-001 — Ubiquitous.** A registry frontend shall keep mutable names,
  versions, tags, channels, and yanks separate from immutable content blobs.
- **FC-REG-002 — Event-driven.** When an artifact is promoted, the FlakeCache
  system shall reuse verified immutable chunks while applying the new roots,
  placement, retention, visibility, signature, and provenance policy.
- **FC-REG-003 — Ubiquitous.** Registry and package adapters shall conform to
  their native protocol rather than emulate all ecosystems through S3.

## Quota, billing, and observability requirements

- **FC-OBS-001 — Ubiquitous.** The FlakeCache system shall meter logical bytes,
  unique physical bytes, requests, transfer, egress, repair, execution, and
  retained versions without double-billing deduplicated chunks.
- **FC-OBS-002 — Event-driven.** When quota or admission limits are exceeded,
  the FlakeCache system shall reject or queue work with a typed reason before
  consuming unbounded storage or execution capacity.
- **FC-OBS-003 — Ubiquitous.** The FlakeCache system shall expose request,
  placement, integrity, repair, capacity, queue, execution, and cost telemetry
  using stable tenant-safe dimensions.
- **FC-OBS-004 — Ubiquitous.** Every credential, policy, placement, retention,
  repair, promotion, and deletion decision shall emit an audit event containing
  actor, tenant, reason, old/new state, revision, and trace identity.

## Specification and execution handover requirements

- **FC-HND-000 — Ubiquitous.** Every accepted need shall identify its versioned
  parent objective and vision, affected stakeholders, observed baseline,
  solution-neutral problem, constraints, decision authority, consequence of no
  action, and falsifier.
- **FC-HND-001 — Ubiquitous.** Every lifecycle handover shall use a versioned,
  content-addressed envelope containing stable change identity, input and output
  digests, producer, accepting authority, validators, evidence, and decision.
- **FC-HND-002 — Event-driven.** When a handover is accepted or rejected, the
  system shall persist the decision, authority, timestamp, policy revision, and
  reason so execution can resume without reconstructing state from chat.
- **FC-HND-003 — Unwanted behavior.** If an envelope is missing, stale,
  unauthorized, unsigned where signatures are required, schema-invalid, or
  lacks mandatory evidence, then the consumer shall reject it before mutation
  or dispatch.
- **FC-HND-004 — Event-driven.** When work fans out, the scheduler shall issue
  capability-scoped child WorkSpecs with disjoint resource leases and shall
  require their evidence digests or an approved waiver before fan-in.
- **FC-HND-005 — Unwanted behavior.** If an agent produced a gate artifact,
  then that same agent shall not be the sole accepting authority for the gate.
- **FC-HND-006 — Optional feature.** Where break-glass operation is enabled,
  the system shall require a named authority, bounded validity window, audit
  record, rollback path, and mandatory retrospective evidence.

## Verification obligations

The implementation is not conformant until evidence covers:

1. YAML and future CNPG catalog schema validation and deterministic replay.
2. Unit/property tests for failure-domain placement and catalog changes.
3. S3 SigV4 and SDK/CLI golden matrices, including supported sccache versions.
4. Existing Nix, WebDAV, GHA, OCI, package, and signing compatibility.
5. Cross-tenant authorization and cache-poisoning adversarial tests.
6. Restart, backend loss, node loss, zone loss, region loss, corruption,
   partition, repair-throttle, rebalance, and recovery drills.
7. Lifecycle, versioning, legal hold, Object Lock, shared-reference, and
   deletion proofs.
8. Remote-execution cancellation, isolation, hermeticity, fan-out, retry, and
   output-idempotency tests.
9. Metering, quotas, billing reconciliation, audit completeness, and dashboard
   checks.
10. Reversible per-cache migration, shadow verification, rollback, and GitOps
    rollout evidence.
11. Handover rejection/resume, supersession, fan-out/fan-in, stale-envelope,
    separation-of-duty, vision-to-need traceability, and break-glass conformance
    evidence.
