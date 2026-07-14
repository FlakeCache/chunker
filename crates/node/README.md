# flakecache-node

The data-plane **node assembly**: ties the core crates into a working
content-addressed, deduplicating store. Zero external dependencies — only our own
crates (`chunker`, `flakecache-cas`, `flakecache-meta`) plus std.

- **`Node::put(bytes)`** — FastCDC-chunks the input, stores each chunk in the CAS
  (idempotent, so identical chunks are stored once = dedup), writes the manifest
  as a CAS object, and records the chunk→manifest→refcount edges in the metadata
  DAG. Returns the manifest `ContentId` (the artifact's address).
- **`Node::get(manifest)`** — looks up the manifest's ordered chunk ids and
  reassembles the exact original bytes (each chunk verified on read).
- **`Node::set_tag`** anchors a GC root; **`Node::collectible_chunks`** reports
  chunks no longer reachable.

This is where dedup is *proven*: two artifacts that share content share chunks,
so the number of unique stored chunks is strictly less than the number of chunk
references. Blob deletion during GC needs a `BlobBackend::remove` (a follow-up);
this crate identifies the collectible set.
