# flakecache-proto-nix

Nix binary-cache HTTP/1.1 protocol front-end for a [`flakecache-node`] `Node`.

Standard-library only: `std::net::TcpListener` + a thread-per-connection loop +
a small hand-written HTTP/1.1 request/response parser. No `tokio`, `hyper`, or
`axum`. It is designed to sit behind an Edge Gateway that terminates TLS and
HTTP/2, so serving plain HTTP/1.1 with `Connection: close` is sufficient.

## Routes

- `GET /nix-cache-info` — returns `StoreDir` / `WantMassQuery` / `Priority`.
- `PUT /<path>` — `node.put(body)`; the returned manifest id is anchored under
  `<path>` as a durable tag so the bytes are retrievable (and GC-rooted).
- `GET /<path>` — resolves `<path>` to its manifest via the tag, reassembles the
  bytes with `node.get`, and returns them (404 if unknown).
- `HEAD /<path>` — existence check (200 / 404), no body.

The path-to-manifest mapping is stored durably in the node's metadata store, so
GETs succeed across process restarts.

[`flakecache-node`]: ../../node
