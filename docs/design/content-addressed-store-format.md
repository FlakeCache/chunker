# FlakeCache content-addressed store format

Status: draft specification

## 1. Purpose and scope

This document specifies the persistent format between FlakeCache's FastCDC
chunker and a pluggable content backend. It defines content identifiers,
manifests, reference counts, commits, witness records, snapshots, recovery, and
garbage collection. It does not specify a store implementation.

The design is adapted from RuVector Format (RVF): its two-level manifests,
`REFCOUNT` segment, append-only writer, SHAKE-256 witness chain, and COW
snapshots. RVF's runtime addresses mutable vector clusters by numeric
`cluster_id`; that runtime is intentionally not reused. Every FlakeCache data
reference defined here is keyed by a content hash.

Normative terms MUST, MUST NOT, SHOULD, and MAY are used as in RFC 2119.

## 2. Layering

```text
NAR byte stream
    |
    v
FastCDC chunker (logical crates/core/chunker; currently this repository root)
    |  ordered (offset, length, chunk bytes)
    v
CAS format in this specification
    |  immutable put(hash, bytes), get(hash)
    v
pluggable backend (filesystem, object storage, or another blob service)
```

The backend contract is deliberately small:

```rust
trait BlobBackend {
    fn put(&self, id: ContentId, bytes: &[u8]) -> Result<()>;
    fn get(&self, id: ContentId) -> Result<Option<Bytes>>;
}
```

`put` MUST be idempotent and MUST reject bytes that do not hash to `id`.
Published blobs are immutable. Tag/reference publication belongs to the CAS
metadata layer above this trait because it requires create-if-absent or atomic
replacement, not content storage.

## 3. Primitive encodings

All integers are unsigned little-endian. Structures are packed wire encodings,
not Rust memory layouts. Reserved bytes MUST be zero when written and ignored
when read unless a later format version assigns them. Variable records are
8-byte aligned with zero padding.

### 3.1 Content identifiers

Version 1 uses a 32-byte SHAKE-256 output:

```text
ContentId = SHAKE256-256(
    "flakecache-cas" || 0x00 ||
    object_kind:u8 || format_version:u16 || payload_length:u64 || payload
)
```

The domain prefix, object kind, version, and length are part of the hash input.
They prevent a byte string from being interpreted as a different object type.
Object kinds are:

| Value | Kind |
| ---: | --- |
| `0x01` | FastCDC chunk |
| `0x02` | Level 0 manifest |
| `0x03` | Level 1 manifest |
| `0x04` | REFCOUNT segment |
| `0x05` | witness segment |
| `0x06` | object recipe/index |
| `0x07` | immutable commit record |

A reader MUST recompute and compare the identifier before using returned blob
bytes. Filesystem implementations SHOULD encode identifiers as lowercase hex
and shard objects as `objects/<first-two-hex>/<remaining-hex>`.

## 4. Two-level manifest

A committed store version has exactly one Level 0 manifest and one referenced
Level 1 manifest. Both are immutable content-addressed blobs.

### 4.1 Level 0: bootstrap manifest

Level 0 is exactly 4096 bytes so a reader can validate and bootstrap the store
with one small read. Unassigned bytes are reserved. The final 32 bytes contain
the SHAKE-256-256 digest of bytes `0x000..0xFDF`, using object kind `0x02`.

| Offset | Size | Field |
| ---: | ---: | --- |
| `0x000` | 4 | magic `FCAS` |
| `0x004` | 2 | format version, `1` |
| `0x006` | 2 | flags |
| `0x008` | 8 | monotonically increasing generation |
| `0x010` | 8 | creation time, nanoseconds since Unix epoch |
| `0x018` | 8 | logical NAR byte length |
| `0x020` | 8 | chunk count |
| `0x028` | 32 | Level 1 `ContentId` |
| `0x048` | 32 | REFCOUNT segment `ContentId`, or zero |
| `0x068` | 32 | witness segment `ContentId`, or zero |
| `0x088` | 32 | witness chain head, or zero |
| `0x0A8` | 32 | parent Level 0 `ContentId`, or zero for root |
| `0x0C8` | 32 | optional Ed25519 public-key identifier |
| `0x0E8` | 64 | optional Ed25519 signature |
| `0x128` | 2 | signature algorithm: `0` none, `1` Ed25519 |
| `0x12A` | 2 | signature length |
| `0x12C` | 3762 | reserved, zero on write |
| `0xFDE` | 2 | bootstrap header length, `0x12C` for v1 |
| `0xFE0` | 32 | Level 0 digest |

The signature, when present, covers the complete Level 0 bytes with the
signature and digest fields zeroed, prefixed by
`"flakecache-level0-signature-v1"`. A signature authenticates a manifest; the
digest alone only detects corruption or tampering.

Readers MUST validate magic, version, reserved requirements for the understood
version, Level 0 digest, and signature when policy requires one before loading
Level 1.

### 4.2 Level 1: full manifest

Level 1 is a variable-length sequence of TLV records sorted by ascending tag.
Each record is:

```text
tag:u16 | flags:u16 | length:u32 | value:[u8; length] | zero padding to 8 bytes
```

Unknown records with flags bit 0 clear MAY be skipped. An unknown record with
flags bit 0 set is critical and MUST cause the reader to reject the manifest.
Singleton tags MUST NOT appear more than once.

| Tag | Name | Cardinality | Value |
| ---: | --- | --- | --- |
| `0x0001` | `OBJECT_TABLE` | one | sorted object recipes |
| `0x0002` | `CHUNK_TABLE` | one | sorted unique chunk descriptors |
| `0x0003` | `PARENT` | zero or one | parent Level 0 `ContentId` |
| `0x0004` | `REFCOUNT` | zero or one | REFCOUNT segment `ContentId` |
| `0x0005` | `WITNESS` | zero or one | witness segment ID and chain head |
| `0x0006` | `METADATA` | zero or one | canonical application metadata |
| `0x0007` | `HASH_ALGORITHMS` | one | algorithm IDs used by the version |

An object recipe reconstructs one logical NAR in stream order. Its fixed header
is `object_id[32] | nar_length:u64 | chunk_count:u64`; it is followed by
`chunk_count` entries of `chunk_id[32] | logical_offset:u64 | length:u32 |
reserved:u32`. Entries MUST be contiguous, non-overlapping, start at offset
zero, and end at `nar_length`. The `object_id` is the hash of the reconstructed
NAR under the caller-selected NAR identity algorithm; that algorithm is named
in `HASH_ALGORITHMS`.

The chunk table is sorted lexicographically by `chunk_id`. Each entry is
`chunk_id[32] | plaintext_length:u64 | stored_length:u64 | codec:u16 |
flags:u16 | reserved:u32`. Chunk boundaries and lengths come from FastCDC;
storage compression or encryption MUST NOT change the logical recipe.

## 5. REFCOUNT segment

REFCOUNT is an immutable, content-addressed snapshot of chunk reachability for
one committed manifest set. Unlike RVF's cluster-indexed array, entries are
keyed by chunk `ContentId`.

Header:

```text
magic[4] = "FCRC"
version:u16 = 1
entry_size:u16 = 40
entry_count:u64
generation:u64
covered_level0_id[32]
reserved[8] = 0
```

The header is followed by `entry_count` entries sorted lexicographically:

```text
chunk_id[32] | refcount:u64
```

Duplicate or zero-count entries are invalid. Counts include references from
all immutable commits reachable from published tags at `generation`. A commit
that changes reachability writes a new REFCOUNT segment; it MUST NOT edit an
old segment in place. The new manifest and its REFCOUNT segment become visible
together at the manifest commit boundary.

REFCOUNT accelerates GC but is repairable derived state. Before deletion, an
implementation MUST confirm zero reachability by traversing all published tag
roots, or by validating that the REFCOUNT generation covers the same complete
tag set. A crash may leak unreferenced blobs; it MUST NOT delete reachable
ones.

## 6. Witness chain

The witness segment is a concatenation of fixed 73-byte entries, matching the
`flakecache-crypto` wire encoding:

```text
prev_hash[32] | action_hash[32] | timestamp_ns:u64 | witness_type:u8
```

The genesis `prev_hash` is zero. Every subsequent `prev_hash` is
SHAKE-256-256 of the complete preceding 73-byte entry. `action_hash` is a
domain-separated hash of the canonical operation record, not an arbitrary log
message. Version 1 reserves these event types:

| Value | Event |
| ---: | --- |
| `0x01` | chunk accepted |
| `0x02` | Level 1 assembled |
| `0x03` | manifest committed |
| `0x04` | tag published |
| `0x05` | tag deleted |
| `0x06` | GC sweep completed |

Level 0 records both the witness segment ID and chain-head hash. Verification
MUST check every link, the head, and each understood action binding. The chain
is tamper-evident but not an identity proof unless the containing Level 0 is
authenticated by a trusted signature.

## 7. Append-only write and commit protocol

All data before step 7 is staged and unreachable. Implementations MUST perform
the steps in this order:

1. Run FastCDC and compute each chunk ID from plaintext chunk bytes.
2. Put missing chunk blobs. Verify an existing blob before treating the put as
   successful.
3. Put object recipe/index blobs and assemble Level 1.
4. Write the new immutable REFCOUNT and witness segments, then Level 1.
5. Write Level 0 referencing the exact IDs from steps 3 and 4.
6. Ensure every referenced blob and containing directory is durable. For a
   filesystem this means file `fsync`, then directory `fsync` as applicable.
7. Publish one immutable commit record and atomically expose its tag binding.

Step 7 is the only atomic visibility flip. A commit record is:

```text
magic[4] = "FCCM" | version:u16 | flags:u16 | generation:u64 |
level0_id[32] | previous_commit_id[32] | created_ns:u64 | reserved[8]
```

The record itself is stored by `ContentId`. A tag binding contains
`tag_length:u16 | tag_utf8 | commit_id[32] | generation:u64`. It is immutable:
once a tag name is bound, it MUST NOT be retargeted. Advancing a branch creates
a new versioned tag (for example `main/00000042`) and MAY atomically replace a
small mutable `HEAD` convenience pointer to that immutable tag.

For local filesystems, publication is write-temp, `fsync(temp)`, atomic rename
or create-if-absent, then `fsync(parent directory)`. A remote metadata service
MUST provide equivalent compare-and-swap semantics. This atomic reference
operation is outside `BlobBackend`.

After a crash, recovery starts only from published tag bindings. It selects the
highest valid generation, verifies the commit and Level 0, and follows content
hashes. Staged or partially written objects that are not reachable from a valid
tag are ignored and may later be collected. Recovery MUST NOT infer a commit
merely because a Level 0 or Level 1 blob exists.

## 8. COW branches and snapshots

Every committed manifest is immutable. A child Level 0 names its parent Level 0
and its Level 1 contains only new object recipes plus replacements or tombstones
for parent entries. Reads resolve an object by searching the child and then its
ancestors. Chunk IDs refer directly to immutable backend blobs, so unchanged
chunks are shared without copying.

A snapshot is an immutable tag-to-commit binding. A branch is a sequence of
immutable versioned tags whose commit records link through
`previous_commit_id`. Creating a branch initially writes only a tag; the first
changed manifest references its parent and stores only changed metadata.
Deleting a tag removes one root from the reachability set but does not mutate
any manifest or chunk.

Implementations SHOULD cap parent traversal depth and periodically emit a
flattened Level 1 manifest. Flattening changes manifest IDs but not chunk IDs.
It MUST be committed through the same append-only protocol.

## 9. Read, verification, and reconstruction

A reader MUST:

1. Resolve a published tag to a commit ID.
2. Fetch and hash-verify the commit, Level 0, and Level 1.
3. Validate the manifest structure and optional required signature.
4. Resolve COW overlays from child to parent, rejecting cycles and generation
   regressions.
5. Fetch each recipe chunk by ID, hash-verify it before decoding, and emit the
   declared plaintext range in order.
6. Verify the reconstructed NAR identity and byte length.

Readers MUST place finite limits on manifest size, TLV count, recipe chunk
count, parent depth, and allocation size before allocating from untrusted
lengths.

## 10. Garbage collection

GC takes a stable snapshot of all published tag roots, traverses commit,
manifest, recipe, chunk, witness, and REFCOUNT references, and marks every
reachable `ContentId`. It may use a matching REFCOUNT generation to accelerate
the mark but not to weaken the reachability check. Sweep removes only unmarked
objects older than the writer grace period. A concurrent writer is safe because
its staged objects are either younger than the grace period or become reachable
through the single atomic commit flip.

## 11. Versioning and deferred algorithms

Version 1 requires SHAKE-256-256 for CAS and witness links and supports optional
Ed25519 manifest signatures. The donor contains no self-contained ML-DSA
implementation, so ML-DSA is not assigned a version 1 algorithm ID. Adding it
requires a separately reviewed dependency, fixed key/signature encodings, test
vectors, and a format-version or algorithm-table extension; readers MUST NOT
interpret an unknown signature algorithm as valid.
