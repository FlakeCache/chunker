# flakecache-backend

Blob backends implementing the CAS [`BlobBackend`] trait.

- **`MemoryBackend`** — an in-process, thread-safe `HashMap` store. For tests and
  ephemeral use.
- **`TieredBackend<Warm, Cold>`** — hot/cold tiering over any two backends:
  - `put` writes **cold first** (the durable system of record), then warm, so a
    completed put is durable even if the node's warm disk is later lost.
  - `get` serves from **warm** when present; on a warm miss it reads **cold** and
    **promotes** (write-back) the bytes into warm so the next read is local.

The `Warm` tier is typically the node's local disk (`flakecache_cas::FilesystemBackend`);
the `Cold` tier is an object/SFTP backend (S3, Hetzner Storage Box) — those cold
implementations are follow-ups; the tiering here works over any `BlobBackend`.
Nothing stores a whole undeduplicated blob: callers store FastCDC chunks by content hash.
