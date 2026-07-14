# flakecache-cas

Content-addressed store for FlakeCache: immutable `put(kind, bytes) -> ContentId`
/ `get(kind, id) -> Option<Bytes>` over a pluggable [`BlobBackend`].

- Content ids are domain-separated SHAKE-256 digests (see
  `docs/design/content-addressed-store-format.md`, §3.1).
- Every read **recomputes and verifies** the identifier before returning bytes.
- `FilesystemBackend` is the local **warm tier**: objects sharded
  `objects/<2hex>/<rest>`, crash-safe temp+rename writes.
- Cold-tier (S3 / Storage Box) write-through and hot/cold tiering layer on top of
  the same `BlobBackend` trait — nothing here stores a whole blob undeduplicated;
  callers store FastCDC chunks addressed by content hash.
