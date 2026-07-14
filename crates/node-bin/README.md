# flakecache-node-bin

The single-node FlakeCache Nix binary-cache server executable.

It constructs a [`flakecache_node::Node`] over a local filesystem CAS plus a redb
metadata store and serves it with the std-only HTTP/1.1 front-end from
[`flakecache-proto-nix`].

## Run

```sh
# defaults: listen 127.0.0.1:8501, data dir ./flakecache-data
flakecache-node-bin

# positional args: <listen-addr> <data-dir>
flakecache-node-bin 0.0.0.0:8501 /var/lib/flakecache

# or via environment
FLAKECACHE_LISTEN=0.0.0.0:8501 FLAKECACHE_DATA_DIR=/var/lib/flakecache flakecache-node-bin
```

Data layout under `<data-dir>`: `cas/objects/...` (content-addressed chunks) and
`meta.redb` (the chunk -> manifest -> refcount DAG and path tags).

Intended to sit behind an Edge Gateway terminating TLS/HTTP2.

[`flakecache-proto-nix`]: ../proto/nix
