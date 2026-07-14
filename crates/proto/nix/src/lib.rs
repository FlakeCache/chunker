// SPDX-License-Identifier: MIT
//! Nix binary-cache HTTP/1.1 front-end for a [`flakecache_node::Node`].
//!
//! A `std`-only server: [`std::net::TcpListener`] with a thread per connection
//! and a small hand-written HTTP/1.1 parser. No async runtime, no `hyper`/`axum`.
//! It is meant to sit behind an Edge Gateway that terminates TLS and HTTP/2, so
//! plain HTTP/1.1 with `Connection: close` (one request per connection) is enough.
//!
//! Routes map directly onto the node:
//! - `GET /nix-cache-info` -> the store's `StoreDir` / `WantMassQuery` / `Priority`.
//! - `PUT /<path>` -> [`Node::put`] the body, then anchor the returned manifest id
//!   under `<path>` via [`Node::set_tag`] so it is retrievable and GC-rooted.
//! - `GET /<path>` -> resolve `<path>` to a manifest via [`Node::get_tag`] and
//!   reassemble the bytes with [`Node::get`] (404 if unknown).
//! - `HEAD /<path>` -> existence check, no body.
//!
//! The path -> manifest mapping lives in the node's durable metadata store, so a
//! GET succeeds even after the process restarts.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

use flakecache_cas::BlobBackend;
use flakecache_node::Node;

/// Reject any request body larger than this (guards against a hostile or bogus
/// `Content-Length` triggering an unbounded allocation). 32 GiB comfortably
/// exceeds any realistic single NAR.
const MAX_BODY_BYTES: u64 = 32 * 1024 * 1024 * 1024;

/// A parsed HTTP/1.1 request: just the pieces this server routes on.
#[derive(Debug)]
struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

/// A Nix binary-cache HTTP server wrapping a [`Node`].
///
/// Construct with [`NixCacheServer::new`], then hand it a bound [`TcpListener`]
/// via [`NixCacheServer::serve`]. The server is shared across connection threads
/// behind an [`Arc`], so `B` must be `Send + Sync`.
#[derive(Debug)]
pub struct NixCacheServer<B: BlobBackend> {
    node: Node<B>,
    store_dir: String,
    priority: u32,
    want_mass_query: bool,
}

impl<B: BlobBackend> NixCacheServer<B> {
    /// Wrap a node with the conventional public-cache defaults (`StoreDir`
    /// `/nix/store`, `WantMassQuery: 1`, `Priority: 30`).
    #[must_use]
    pub fn new(node: Node<B>) -> Self {
        Self {
            node,
            store_dir: "/nix/store".to_owned(),
            priority: 30,
            want_mass_query: true,
        }
    }

    /// Override the advertised `StoreDir`.
    #[must_use]
    pub fn with_store_dir(mut self, store_dir: impl Into<String>) -> Self {
        self.store_dir = store_dir.into();
        self
    }

    /// Override the advertised cache `Priority` (lower wins in Nix's ordering).
    #[must_use]
    pub const fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Borrow the wrapped node.
    #[must_use]
    pub const fn node(&self) -> &Node<B> {
        &self.node
    }

    /// The `/nix-cache-info` response body.
    fn cache_info(&self) -> String {
        format!(
            "StoreDir: {}\nWantMassQuery: {}\nPriority: {}\n",
            self.store_dir,
            u8::from(self.want_mass_query),
            self.priority
        )
    }
}

impl<B: BlobBackend + Send + Sync + 'static> NixCacheServer<B> {
    /// Accept connections on `listener` forever, handling each on its own thread.
    ///
    /// Returns only if accepting a connection fails fatally; per-connection errors
    /// are logged to stderr and do not stop the server.
    ///
    /// # Errors
    /// Returns [`io::Error`] if [`TcpListener::accept`] itself fails.
    pub fn serve(self: Arc<Self>, listener: &TcpListener) -> io::Result<()> {
        for stream in listener.incoming() {
            let stream = stream?;
            let server = Arc::clone(&self);
            let _ = thread::Builder::new()
                .name("nix-cache-conn".to_owned())
                .spawn(move || server.handle_connection(&stream));
        }
        Ok(())
    }

    /// Handle a single connection: parse one request, route it, write one response.
    fn handle_connection(&self, stream: &TcpStream) {
        let outcome = match read_request(stream) {
            Ok(Some(request)) => self.route(&request, stream),
            // Clean EOF before any request line: nothing to do.
            Ok(None) => Ok(()),
            Err(HttpError::BadRequest(msg)) => write_response(
                stream,
                400,
                "Bad Request",
                "text/plain",
                msg.as_bytes(),
                true,
            ),
            Err(HttpError::Io(err)) => Err(err),
        };
        if let Err(err) = outcome {
            eprintln!("nix-cache: connection error: {err}");
        }
    }

    /// Dispatch a parsed request to the matching handler.
    fn route(&self, request: &Request, stream: &TcpStream) -> io::Result<()> {
        match (request.method.as_str(), request.path.as_str()) {
            ("GET", "/nix-cache-info") => {
                let body = self.cache_info();
                write_response(
                    stream,
                    200,
                    "OK",
                    "text/x-nix-cache-info",
                    body.as_bytes(),
                    true,
                )
            }
            ("GET", path) => self.handle_get(path, stream, true),
            ("HEAD", "/nix-cache-info") => {
                write_response(stream, 200, "OK", "text/x-nix-cache-info", b"", false)
            }
            ("HEAD", path) => self.handle_get(path, stream, false),
            ("PUT", path) => self.handle_put(path, &request.body, stream),
            (method, _) => {
                let msg = format!("method {method} not allowed\n");
                write_response(
                    stream,
                    405,
                    "Method Not Allowed",
                    "text/plain",
                    msg.as_bytes(),
                    true,
                )
            }
        }
    }

    /// `GET`/`HEAD` `<path>`: resolve the tag to a manifest and reassemble.
    fn handle_get(&self, path: &str, stream: &TcpStream, with_body: bool) -> io::Result<()> {
        let manifest = match self.node.get_tag(path) {
            Ok(Some(id)) => id,
            Ok(None) => {
                return write_response(stream, 404, "Not Found", "text/plain", b"", with_body);
            }
            Err(err) => return server_error(stream, &err, with_body),
        };
        match self.node.get(manifest) {
            Ok(Some(bytes)) => write_response(
                stream,
                200,
                "OK",
                "application/octet-stream",
                &bytes,
                with_body,
            ),
            // Tagged but the manifest vanished: treat as absent.
            Ok(None) => write_response(stream, 404, "Not Found", "text/plain", b"", with_body),
            Err(err) => server_error(stream, &err, with_body),
        }
    }

    /// `PUT <path>`: ingest the body and anchor it under `<path>`.
    fn handle_put(&self, path: &str, body: &[u8], stream: &TcpStream) -> io::Result<()> {
        let manifest = match self.node.put(body) {
            Ok(id) => id,
            Err(err) => return server_error(stream, &err, true),
        };
        if let Err(err) = self.node.set_tag(path, manifest) {
            return server_error(stream, &err, true);
        }
        write_response(stream, 200, "OK", "text/plain", b"", true)
    }
}

/// Write a 500 response for an internal node error (also logged to stderr).
fn server_error<E: std::fmt::Display>(
    stream: &TcpStream,
    err: &E,
    with_body: bool,
) -> io::Result<()> {
    eprintln!("nix-cache: internal error: {err}");
    write_response(
        stream,
        500,
        "Internal Server Error",
        "text/plain",
        b"internal error\n",
        with_body,
    )
}

/// A failure while reading a request.
#[derive(Debug)]
enum HttpError {
    /// The request was malformed -> 400.
    BadRequest(String),
    /// A socket read failed.
    Io(io::Error),
}

impl From<io::Error> for HttpError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

/// Read and parse a single HTTP/1.1 request from `stream`.
///
/// Returns `Ok(None)` on a clean EOF before any request line (the client opened
/// and closed a connection). The body is read from the *same* buffered reader as
/// the head, so already-buffered body bytes are not lost.
fn read_request(stream: &TcpStream) -> Result<Option<Request>, HttpError> {
    let mut reader = BufReader::new(stream);

    // Request line: METHOD SP TARGET SP VERSION.
    let mut line = String::new();
    let read = reader.read_line(&mut line)?;
    if read == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let (Some(method), Some(path)) = (parts.next(), parts.next()) else {
        return Err(HttpError::BadRequest("malformed request line\n".to_owned()));
    };
    let method = method.to_owned();
    let path = path.to_owned();

    // Headers until a blank line; we only need Content-Length.
    let mut content_length: u64 = 0;
    loop {
        let mut header = String::new();
        let n = reader.read_line(&mut header)?;
        if n == 0 {
            return Err(HttpError::BadRequest(
                "unexpected eof in headers\n".to_owned(),
            ));
        }
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            if name.trim().eq_ignore_ascii_case("content-length") {
                content_length = value
                    .trim()
                    .parse()
                    .map_err(|_| HttpError::BadRequest("invalid content-length\n".to_owned()))?;
            }
        }
    }

    if content_length > MAX_BODY_BYTES {
        return Err(HttpError::BadRequest("request body too large\n".to_owned()));
    }
    let len = usize::try_from(content_length)
        .map_err(|_| HttpError::BadRequest("content-length out of range\n".to_owned()))?;
    let mut body = vec![0_u8; len];
    reader.read_exact(&mut body)?;

    Ok(Some(Request { method, path, body }))
}

/// Write one HTTP/1.1 response and close (`Connection: close`).
///
/// `Content-Length` is always set (to the true body length even when the body is
/// suppressed for a HEAD, per RFC 9110), so a client can frame the response
/// deterministically.
fn write_response(
    stream: &TcpStream,
    status: u16,
    reason: &str,
    content_type: &str,
    body: &[u8],
    with_body: bool,
) -> io::Result<()> {
    let mut out = stream;
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Length: {}\r\n\
         Content-Type: {content_type}\r\n\
         Connection: close\r\n\
         \r\n",
        body.len()
    );
    out.write_all(head.as_bytes())?;
    if with_body {
        out.write_all(body)?;
    }
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use flakecache_cas::{Cas, FilesystemBackend};
    use flakecache_meta::MetaStore;
    use flakecache_node::Node;
    use std::io::{Read, Write};

    /// A minimal std-only HTTP client: send one request, return (status, body).
    fn request(
        addr: std::net::SocketAddr,
        method: &str,
        path: &str,
        body: &[u8],
    ) -> (u16, Vec<u8>) {
        let mut stream = TcpStream::connect(addr).unwrap();
        let head = format!(
            "{method} {path} HTTP/1.1\r\nHost: test\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        stream.write_all(head.as_bytes()).unwrap();
        stream.write_all(body).unwrap();
        stream.flush().unwrap();

        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).unwrap();

        // Split head/body on the CRLFCRLF terminator.
        let split = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .expect("response has header terminator");
        let head = String::from_utf8(raw[..split].to_vec()).unwrap();
        let resp_body = raw[split + 4..].to_vec();
        let status: u16 = head
            .lines()
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        (status, resp_body)
    }

    fn spawn_server() -> (std::net::SocketAddr, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let cas = Cas::new(FilesystemBackend::new(dir.path().join("cas")));
        let meta = MetaStore::open(dir.path().join("meta.redb")).unwrap();
        let node = Node::new(cas, meta);
        let server = Arc::new(NixCacheServer::new(node));

        // Bind first, capture the ephemeral port, THEN move the bound listener
        // into the serve thread -- connections queue on a bound socket, so this
        // removes the accept race without any sleep.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let _ = thread::spawn(move || server.serve(&listener));
        (addr, dir)
    }

    #[test]
    fn put_then_get_round_trips_bytes_exactly() {
        let (addr, _dir) = spawn_server();
        let payload: Vec<u8> = (0..250_000_u32).flat_map(u32::to_le_bytes).collect();

        let (put_status, _) = request(addr, "PUT", "/nar/roundtrip.nar", &payload);
        assert_eq!(put_status, 200, "PUT should succeed");

        let (get_status, got) = request(addr, "GET", "/nar/roundtrip.nar", b"");
        assert_eq!(get_status, 200, "GET of a stored path should succeed");
        assert_eq!(got, payload, "GET must return the exact PUT bytes");
    }

    #[test]
    fn get_unknown_path_is_404() {
        let (addr, _dir) = spawn_server();
        let (status, _) = request(addr, "GET", "/nar/never-uploaded.nar", b"");
        assert_eq!(status, 404);
    }

    #[test]
    fn nix_cache_info_is_served() {
        let (addr, _dir) = spawn_server();
        let (status, body) = request(addr, "GET", "/nix-cache-info", b"");
        assert_eq!(status, 200);
        let text = String::from_utf8(body).unwrap();
        assert!(text.contains("StoreDir: /nix/store"), "got: {text}");
        assert!(text.contains("WantMassQuery: 1"), "got: {text}");
        assert!(text.contains("Priority: 30"), "got: {text}");
    }

    #[test]
    fn head_reports_existence_without_body() {
        let (addr, _dir) = spawn_server();
        let (miss, _) = request(addr, "HEAD", "/x.narinfo", b"");
        assert_eq!(miss, 404, "HEAD of an absent path is 404");

        let (put, _) = request(addr, "PUT", "/x.narinfo", b"StorePath: /nix/store/xyz\n");
        assert_eq!(put, 200);
        let (hit, body) = request(addr, "HEAD", "/x.narinfo", b"");
        assert_eq!(hit, 200, "HEAD of a present path is 200");
        assert!(body.is_empty(), "HEAD returns no body");
    }
}
