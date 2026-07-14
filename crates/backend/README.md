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
the durable cold tier can be `S3Backend`, configured with
`FLAKECACHE_S3_ENDPOINT`, `FLAKECACHE_S3_BUCKET`, `FLAKECACHE_S3_REGION`,
`FLAKECACHE_S3_ACCESS_KEY_ID`, `FLAKECACHE_S3_SECRET_ACCESS_KEY`, and optional
`FLAKECACHE_S3_PREFIX`. S3 requests use Garage-compatible path-style URLs,
rustls, and AWS SigV4. The tiering remains generic over any `BlobBackend`, so a
Hetzner Storage Box implementation can be added independently. Nothing stores a
whole undeduplicated blob: callers store FastCDC chunks by content hash.
