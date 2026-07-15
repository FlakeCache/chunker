// SPDX-License-Identifier: MIT
//! HTTP peer transport for the swarm.
//!
//! [`HttpPeerClient`] implements [`PeerClient`] over a small synchronous
//! `rustls` client (the same `ureq` stack the S3 cold tier uses), so a
//! [`crate::Router`] can fetch a chunk from, and replicate a chunk to, its
//! co-owner peers.
//!
//! ## Addressing
//!
//! A [`NodeId`] used with this client is the peer's **base URL**, for example
//! `http://cache-2.internal:8501` (no trailing slash). A blob with content id
//! `<hex>` is transferred at `<node>/swarm/blob/<hex>`:
//! - `GET`  -> `200` with the raw bytes, or `404` if the peer lacks the blob.
//! - `PUT`  -> store the body (idempotent, immutable content).
//!
//! The receiving `/swarm/blob/<hex>` endpoint is a later increment, so swarm
//! mode is **not end-to-end yet** and must not be enabled on a real node. The
//! current Nix-cache front-end has no such route: a `GET` resolves no tag and
//! returns `404` (a harmless read miss), but a `PUT` is either rejected with
//! `401`/`403` when write-auth is on, or — with auth off — accepted by the
//! catch-all Nix `PUT` handler and *misingested* as a cache object under a bogus
//! tag. Do not set `FLAKECACHE_SWARM_SELF` in production until the receive
//! endpoint exists; the wiring is present for tests and the follow-up increment.

use std::time::Duration;

use bytes::Bytes;
use flakecache_cas::ContentId;

use crate::NodeId;
use crate::router::{PeerClient, PeerError};

/// Default per-call timeout for a peer request, bounding how long a single
/// fetch or replica push may stall before it is treated as a failure.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

/// Cap on a peer response body read into memory. CAS blobs are individual
/// chunks/manifests well under this bound; it exists only to refuse a hostile
/// or corrupt peer streaming an unbounded body.
const MAX_BLOB_BYTES: u64 = 256 * 1024 * 1024;

/// A [`PeerClient`] that transfers blobs over HTTP to peer base URLs.
#[derive(Debug, Clone)]
pub struct HttpPeerClient {
    agent: ureq::Agent,
}

impl HttpPeerClient {
    /// Create a client whose fetches and pushes each time out after `timeout`.
    #[must_use]
    pub fn new(timeout: Duration) -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            .build();
        Self {
            agent: ureq::Agent::new_with_config(config),
        }
    }

    /// The `<node>/swarm/blob/<hex>` URL for `id` on `node`.
    fn blob_url(node: &NodeId, id: ContentId) -> String {
        format!("{}/swarm/blob/{}", node.as_str(), id.to_hex())
    }
}

impl Default for HttpPeerClient {
    fn default() -> Self {
        Self::new(DEFAULT_TIMEOUT)
    }
}

impl PeerClient for HttpPeerClient {
    fn fetch(&self, node: &NodeId, id: ContentId) -> Result<Option<Bytes>, PeerError> {
        let url = Self::blob_url(node, id);
        match self.agent.get(&url).call() {
            Ok(mut response) => {
                let bytes = response
                    .body_mut()
                    .with_config()
                    .limit(MAX_BLOB_BYTES)
                    .read_to_vec()
                    .map_err(|e| Box::new(e) as PeerError)?;
                Ok(Some(Bytes::from(bytes)))
            }
            Err(ureq::Error::StatusCode(404)) => Ok(None),
            Err(error) => Err(Box::new(error) as PeerError),
        }
    }

    fn push(&self, node: &NodeId, id: ContentId, bytes: &[u8]) -> Result<(), PeerError> {
        let url = Self::blob_url(node, id);
        self.agent
            .put(&url)
            .send(bytes)
            .map(|_| ())
            .map_err(|e| Box::new(e) as PeerError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flakecache_cas::ObjectKind;
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Arc, Mutex};
    use std::thread;

    /// A minimal in-repo HTTP peer that speaks the `/swarm/blob/<hex>` protocol
    /// against a real bound socket: `PUT` stores the body, `GET` returns it or a
    /// `404`. One request per connection, explicit `Content-Length`, so the
    /// `ureq` client parses a well-framed response and never hangs.
    struct TestPeer {
        base_url: String,
        blobs: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    }

    impl TestPeer {
        fn start() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind test peer");
            let base_url = format!("http://{}", listener.local_addr().expect("addr"));
            let blobs: Arc<Mutex<HashMap<String, Vec<u8>>>> = Arc::default();
            let store = Arc::clone(&blobs);
            thread::spawn(move || {
                for stream in listener.incoming() {
                    let Ok(stream) = stream else { break };
                    handle(&stream, &store);
                }
            });
            Self { base_url, blobs }
        }

        fn node(&self) -> NodeId {
            NodeId::new(self.base_url.clone())
        }

        fn contains_hex(&self, hex: &str) -> bool {
            self.blobs
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .contains_key(hex)
        }
    }

    fn handle(mut stream: &TcpStream, store: &Mutex<HashMap<String, Vec<u8>>>) {
        let mut reader = BufReader::new(stream);
        let mut request_line = String::new();
        if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
            return;
        }
        let mut parts = request_line.split_whitespace();
        let method = parts.next().unwrap_or_default().to_owned();
        let path = parts.next().unwrap_or_default().to_owned();
        let hex = path.rsplit('/').next().unwrap_or_default().to_owned();

        // Consume headers, capturing the body length for a PUT.
        let mut content_length = 0_usize;
        loop {
            let mut header = String::new();
            if reader.read_line(&mut header).unwrap_or(0) == 0 {
                break;
            }
            let header = header.trim_end();
            if header.is_empty() {
                break;
            }
            if let Some(value) = header.to_ascii_lowercase().strip_prefix("content-length:") {
                content_length = value.trim().parse().unwrap_or(0);
            }
        }

        match method.as_str() {
            "PUT" => {
                let mut body = vec![0_u8; content_length];
                reader.read_exact(&mut body).expect("read PUT body");
                store
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(hex, body);
                write_response(&mut stream, "200 OK", b"");
            }
            "GET" => {
                let body = store
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get(&hex)
                    .cloned();
                match body {
                    Some(bytes) => write_response(&mut stream, "200 OK", &bytes),
                    None => write_response(&mut stream, "404 Not Found", b""),
                }
            }
            _ => write_response(&mut stream, "405 Method Not Allowed", b""),
        }
    }

    fn write_response(stream: &mut &TcpStream, status: &str, body: &[u8]) {
        let header = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len(),
        );
        let _ = stream.write_all(header.as_bytes());
        let _ = stream.write_all(body);
        let _ = stream.flush();
    }

    fn cid(seed: &[u8]) -> ContentId {
        ContentId::compute(ObjectKind::Chunk, seed)
    }

    #[test]
    fn push_then_fetch_round_trips_over_http() {
        let peer = TestPeer::start();
        let client = HttpPeerClient::new(DEFAULT_TIMEOUT);
        let id = cid(b"payload");

        client
            .push(&peer.node(), id, b"payload")
            .expect("push succeeds");
        assert!(peer.contains_hex(&id.to_hex()), "peer stored the blob");

        let fetched = client.fetch(&peer.node(), id).expect("fetch succeeds");
        assert_eq!(fetched.as_deref(), Some(&b"payload"[..]));
    }

    #[test]
    fn fetch_absent_blob_is_none() {
        let peer = TestPeer::start();
        let client = HttpPeerClient::new(DEFAULT_TIMEOUT);
        let fetched = client
            .fetch(&peer.node(), cid(b"missing"))
            .expect("a 404 is Ok(None), not an error");
        assert!(fetched.is_none());
    }

    #[test]
    fn fetch_from_unreachable_peer_is_error() {
        let client = HttpPeerClient::new(Duration::from_millis(200));
        // Reserved-for-documentation address that does not accept connections.
        let node = NodeId::new("http://192.0.2.1:9");
        let result = client.fetch(&node, cid(b"x"));
        assert!(result.is_err(), "a transport failure surfaces as an error");
    }
}
