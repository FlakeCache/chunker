# flakecache-meta

The node-local metadata engine: a disk-resident **chunk → manifest → refcount
DAG** with mark-and-sweep garbage-collection reachability.

- Backed by [`redb`] (a crash-safe, mmap'd, on-disk B-tree) — the working set is
  paged by the OS, so the index is **not RAM-bound**.
- `put_manifest` records a manifest as an ordered list of chunk `ContentId`s and
  maintains per-chunk **refcounts** (idempotent; shared chunks dedup).
- GC is exact **mark-and-sweep**: `reachable_chunks` marks every chunk reachable
  from a live root manifest; `collectible_chunks` returns tracked chunks that are
  *not* reachable (the caller deletes those blobs from the CAS).

Roots (tags) are the GC anchors. Refcounts are maintained as an O(1) stat and the
basis for future CRDT/PN-counter convergence in the swarm; reachability is the
authority for what may be collected.
