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
//! - With an auth key configured, `PUT`, `DELETE`, and `MKCOL` require a valid
//!   Ed25519 JWT carrying `scope: "write"`; read routes remain anonymous.
//!
//! The path -> manifest mapping lives in the node's durable metadata store, so a
//! GET succeeds even after the process restarts.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use flakecache_cas::BlobBackend;
use flakecache_crypto::{SigningKey, VerifyingKey, verify_message};
use flakecache_node::Node;

pub mod narinfo;

use narinfo::NarInfo;

/// Reject any request body larger than this (guards against a hostile or bogus
/// `Content-Length` triggering an unbounded allocation). 32 GiB comfortably
/// exceeds any realistic single NAR.
const MAX_BODY_BYTES: u64 = 32 * 1024 * 1024 * 1024;

/// A parsed HTTP/1.1 request: just the pieces this server routes on.
#[derive(Debug)]
struct Request {
    method: String,
    path: String,
    authorization: Option<String>,
    body: Vec<u8>,
}

/// A Nix binary-cache HTTP server wrapping a [`Node`].
///
/// Construct with [`NixCacheServer::new`], then hand it a bound [`TcpListener`]
/// via [`NixCacheServer::serve`]. The server is shared across connection threads
/// behind an [`Arc`], so `B` must be `Send + Sync`.
pub struct NixCacheServer<B: BlobBackend> {
    node: Node<B>,
    store_dir: String,
    priority: u32,
    want_mass_query: bool,
    /// The cache's `(key-name, secret-key)`; when set, narinfos are (re)signed
    /// with it on `GET` so Nix clients trusting the matching public key
    /// substitute. `None` serves stored narinfos verbatim.
    signing: Option<(String, SigningKey)>,
    auth_key: Option<VerifyingKey>,
}

// Manual `Debug` so the Ed25519 secret key is never rendered.
impl<B: BlobBackend> std::fmt::Debug for NixCacheServer<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NixCacheServer")
            .field("store_dir", &self.store_dir)
            .field("priority", &self.priority)
            .field("want_mass_query", &self.want_mass_query)
            .field("signing_key", &self.signing.as_ref().map(|(name, _)| name))
            .field("auth_key", &self.auth_key.as_ref().map(|_| "configured"))
            .finish_non_exhaustive()
    }
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
            signing: None,
            auth_key: None,
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

    /// Sign served narinfos with this cache key (`key_name` + Ed25519 secret).
    ///
    /// With a key set, `GET /<hash>.narinfo` returns a narinfo bearing a `Sig:`
    /// line under `key_name`, so a Nix client that trusts the matching public key
    /// will substitute. Parse [`narinfo::parse_secret_key`] from a Nix
    /// `<name>:<base64>` secret key.
    #[must_use]
    pub fn with_signing_key(mut self, key_name: impl Into<String>, secret_key: SigningKey) -> Self {
        self.signing = Some((key_name.into(), secret_key));
        self
    }

    /// Require valid write-scoped capability tokens for mutating HTTP methods.
    #[must_use]
    pub fn with_auth_key(mut self, key: VerifyingKey) -> Self {
        self.auth_key = Some(key);
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
        if std::env::var_os("FLAKECACHE_ACCESS_LOG").is_some() {
            eprintln!(
                "nix-cache: {} {} (body {}B)",
                request.method,
                request.path,
                request.body.len()
            );
        }
        if matches!(request.method.as_str(), "PUT" | "DELETE" | "MKCOL") {
            match self.authorize_write(request.authorization.as_deref()) {
                Ok(()) => {}
                Err(AuthError::Unauthorized) => {
                    return write_response(
                        stream,
                        401,
                        "Unauthorized",
                        "text/plain",
                        b"unauthorized\n",
                        true,
                    );
                }
                Err(AuthError::Forbidden) => {
                    return write_response(
                        stream,
                        403,
                        "Forbidden",
                        "text/plain",
                        b"forbidden\n",
                        true,
                    );
                }
            }
        }
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
            // Minimal WebDAV verbs so an sccache/opendal WebDAV client can use
            // this as a compilation-cache backend. The store is flat and
            // content-keyed, so collections are implicit: MKCOL is a no-op and
            // PROPFIND reports any path as an existing collection.
            ("OPTIONS", _) => write_dav_options(stream),
            ("MKCOL", _) => write_response(stream, 201, "Created", "text/plain", b"", true),
            ("PROPFIND", path) => Self::handle_propfind(path, stream),
            ("DELETE", path) => self.handle_delete(path, stream),
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

    fn authorize_write(&self, authorization: Option<&str>) -> Result<(), AuthError> {
        let Some(key) = self.auth_key.as_ref() else {
            return Ok(());
        };
        let token = authorization
            .and_then(|value| value.strip_prefix("Bearer "))
            .ok_or(AuthError::Unauthorized)?;
        let claims = verify_token(token, key).ok_or(AuthError::Unauthorized)?;
        if claims.scope == "write" {
            Ok(())
        } else {
            Err(AuthError::Forbidden)
        }
    }

    /// `GET`/`HEAD` `<path>`: resolve the tag to a manifest and reassemble.
    ///
    /// A `.narinfo` path is (re)signed with the cache key when one is configured,
    /// so the client receives a signature it trusts; everything else (the NARs) is
    /// returned verbatim.
    fn handle_get(&self, path: &str, stream: &TcpStream, with_body: bool) -> io::Result<()> {
        let manifest = match self.node.get_tag(path) {
            Ok(Some(id)) => id,
            Ok(None) => {
                return write_response(stream, 404, "Not Found", "text/plain", b"", with_body);
            }
            Err(err) => return server_error(stream, &err, with_body),
        };
        let bytes = match self.node.get(manifest) {
            Ok(Some(bytes)) => bytes,
            // Tagged but the manifest vanished: treat as absent.
            Ok(None) => {
                return write_response(stream, 404, "Not Found", "text/plain", b"", with_body);
            }
            Err(err) => return server_error(stream, &err, with_body),
        };

        if path.ends_with(".narinfo") {
            let signed = self.signed_narinfo(&bytes);
            let body = signed.as_deref().unwrap_or(&bytes);
            return write_response(stream, 200, "OK", "text/x-nix-narinfo", body, with_body);
        }
        write_response(
            stream,
            200,
            "OK",
            "application/octet-stream",
            &bytes,
            with_body,
        )
    }

    /// Re-sign a stored narinfo with the cache key.
    ///
    /// Returns `None` (serve the stored bytes verbatim) when no key is configured
    /// or the stored bytes are not a parseable narinfo — e.g. an already-signed
    /// zero-knowledge upload we must not rewrite.
    fn signed_narinfo(&self, stored: &[u8]) -> Option<Vec<u8>> {
        let (name, key) = self.signing.as_ref()?;
        let text = std::str::from_utf8(stored).ok()?;
        let (narinfo, _existing_sigs) = NarInfo::parse(text, &self.store_dir).ok()?;
        let sig = narinfo.signature(name, key);
        Some(narinfo.to_text(&[sig]).into_bytes())
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

    /// `PROPFIND <path>`: report the path as an existing `WebDAV` collection.
    ///
    /// The store is flat and content-keyed, so every path is treated as an
    /// existing collection; this satisfies an `opendal` `WebDAV` client's
    /// directory `stat` that gates cache-object writes.
    fn handle_propfind(path: &str, stream: &TcpStream) -> io::Result<()> {
        let name = path.trim_end_matches('/').rsplit('/').next().unwrap_or("");
        // Mirror the exact multistatus shape opendal's WebDAV client tests parse
        // (displayname + getlastmodified + resourcetype), so its write-check stat
        // succeeds instead of failing to deserialize.
        let body = format!(
            "<D:multistatus xmlns:D=\"DAV:\">\n\
             <D:response>\n\
             <D:href>{path}</D:href>\n\
             <D:propstat>\n\
             <D:prop>\n\
             <D:displayname>{name}</D:displayname>\n\
             <D:getlastmodified>Thu, 01 Jan 1970 00:00:00 GMT</D:getlastmodified>\n\
             <D:resourcetype><D:collection/></D:resourcetype>\n\
             </D:prop>\n\
             <D:status>HTTP/1.1 200 OK</D:status>\n\
             </D:propstat>\n\
             </D:response>\n\
             </D:multistatus>\n"
        );
        write_response(
            stream,
            207,
            "Multi-Status",
            "application/xml; charset=utf-8",
            body.as_bytes(),
            true,
        )
    }

    /// `DELETE <path>`: drop the path->manifest mapping (best-effort; the blob is
    /// reclaimed by `GC`, not here). Idempotent: always 204.
    fn handle_delete(&self, path: &str, stream: &TcpStream) -> io::Result<()> {
        if let Err(err) = self.node.meta().remove_root(path) {
            return server_error(stream, &err, false);
        }
        write_response(stream, 204, "No Content", "text/plain", b"", false)
    }
}

/// Respond to a `WebDAV` `OPTIONS` probe advertising the verbs this cache supports.
fn write_dav_options(stream: &TcpStream) -> io::Result<()> {
    let mut out = stream;
    let head = "HTTP/1.1 200 OK\r\n\
                Content-Length: 0\r\n\
                DAV: 1,2\r\n\
                Allow: OPTIONS, GET, HEAD, PUT, DELETE, MKCOL, PROPFIND\r\n\
                Connection: close\r\n\r\n";
    out.write_all(head.as_bytes())?;
    out.flush()
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

#[derive(Debug, PartialEq, Eq)]
enum AuthError {
    Unauthorized,
    Forbidden,
}

struct TokenClaims {
    scope: String,
}

fn verify_token(token: &str, key: &VerifyingKey) -> Option<TokenClaims> {
    let mut segments = token.split('.');
    let header_segment = segments.next()?;
    let payload_segment = segments.next()?;
    let signature_segment = segments.next()?;
    if segments.next().is_some() {
        return None;
    }

    let header = String::from_utf8(decode_base64url(header_segment)?).ok()?;
    let payload = String::from_utf8(decode_base64url(payload_segment)?).ok()?;
    let signature: [u8; 64] = decode_base64url(signature_segment)?.try_into().ok()?;
    let signed = format!("{header_segment}.{payload_segment}");
    if !verify_message(signed.as_bytes(), &signature, key) {
        return None;
    }

    let header = JsonObject::parse(&header)?;
    if header.string("alg")? != "EdDSA" {
        return None;
    }
    let payload = JsonObject::parse(&payload)?;
    let _cache = payload.string("cache")?;
    let scope = payload.string("scope")?.to_owned();
    if scope != "read" && scope != "write" {
        return None;
    }
    let exp = u64::try_from(payload.integer("exp")?).ok()?;
    let _iat = payload.integer("iat")?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();
    (exp > now).then_some(TokenClaims { scope })
}

fn decode_base64url(input: &str) -> Option<Vec<u8>> {
    if input.contains('=') {
        return None;
    }
    let mut standard = input.replace('-', "+").replace('_', "/");
    match standard.len() % 4 {
        0 => {}
        2 => standard.push_str("=="),
        3 => standard.push('='),
        _ => return None,
    }
    flakecache_crypto::b64::decode(&standard).ok()
}

struct JsonObject<'a> {
    source: &'a str,
}

impl<'a> JsonObject<'a> {
    fn parse(source: &'a str) -> Option<Self> {
        let source = source.trim();
        (source.starts_with('{') && source.ends_with('}')).then_some(Self { source })
    }

    fn string(&self, key: &str) -> Option<&'a str> {
        let value = self.value(key)?.strip_prefix('"')?;
        let end = value.find('"')?;
        let string = &value[..end];
        (!string.contains('\\')).then_some(string)
    }

    fn integer(&self, key: &str) -> Option<i64> {
        let value = self.value(key)?;
        let end = value
            .find(|character: char| !character.is_ascii_digit() && character != '-')
            .unwrap_or(value.len());
        value[..end].parse().ok()
    }

    fn value(&self, key: &str) -> Option<&'a str> {
        let needle = format!("\"{key}\"");
        let mut rest = self.source.get(1..self.source.len() - 1)?;
        loop {
            rest = rest.trim_start();
            if rest.starts_with(&needle) {
                let after_key = rest.get(needle.len()..)?.trim_start();
                return after_key.strip_prefix(':').map(str::trim_start);
            }
            let comma = find_json_comma(rest)?;
            rest = rest.get(comma + 1..)?;
        }
    }
}

fn find_json_comma(source: &str) -> Option<usize> {
    let mut quoted = false;
    let mut escaped = false;
    let mut depth = 0_u32;
    for (index, byte) in source.bytes().enumerate() {
        if quoted && byte == b'\\' && !escaped {
            escaped = true;
            continue;
        }
        if byte == b'"' && !escaped {
            quoted = !quoted;
        } else if !quoted {
            match byte {
                b'{' | b'[' => depth += 1,
                b'}' | b']' => depth = depth.saturating_sub(1),
                b',' if depth == 0 => return Some(index),
                _ => {}
            }
        }
        escaped = false;
    }
    None
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
    let mut authorization = None;
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
            } else if name.trim().eq_ignore_ascii_case("authorization") {
                authorization = Some(value.trim().to_owned());
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

    Ok(Some(Request {
        method,
        path,
        authorization,
        body,
    }))
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
        request_with_auth(addr, method, path, body, None)
    }

    fn request_with_auth(
        addr: std::net::SocketAddr,
        method: &str,
        path: &str,
        body: &[u8],
        token: Option<&str>,
    ) -> (u16, Vec<u8>) {
        let mut stream = TcpStream::connect(addr).unwrap();
        let authorization = token.map_or_else(String::new, |token| {
            format!("Authorization: Bearer {token}\r\n")
        });
        let head = format!(
            "{method} {path} HTTP/1.1\r\nHost: test\r\nContent-Length: {}\r\n{authorization}Connection: close\r\n\r\n",
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

    fn spawn_auth_server(key: VerifyingKey) -> (std::net::SocketAddr, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let cas = Cas::new(FilesystemBackend::new(dir.path().join("cas")));
        let meta = MetaStore::open(dir.path().join("meta.redb")).unwrap();
        let node = Node::new(cas, meta);
        let server = Arc::new(NixCacheServer::new(node).with_auth_key(key));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let _ = thread::spawn(move || server.serve(&listener));
        (addr, dir)
    }

    fn base64url(input: &[u8]) -> String {
        flakecache_crypto::b64::encode(input)
            .replace('+', "-")
            .replace('/', "_")
            .trim_end_matches('=')
            .to_owned()
    }

    fn token(key: &SigningKey, scope: &str, exp: u64) -> String {
        let header = base64url(br#"{"alg":"EdDSA","typ":"JWT"}"#);
        let payload = base64url(
            format!(r#"{{"cache":"default","extra":{{"ignored":true}},"scope":"{scope}","exp":{exp},"iat":1}}"#).as_bytes(),
        );
        let signed = format!("{header}.{payload}");
        let signature = flakecache_crypto::sign_message(signed.as_bytes(), key);
        format!("{signed}.{}", base64url(&signature))
    }

    fn future_exp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 60
    }

    #[test]
    fn valid_write_token_allows_put() {
        let key = flakecache_crypto::signing_key_from_seed(&[11; 32]);
        let (addr, _dir) = spawn_auth_server(key.verifying_key());
        let token = token(&key, "write", future_exp());
        assert_eq!(
            request_with_auth(addr, "PUT", "/authorized", b"data", Some(&token)).0,
            200
        );
    }

    #[test]
    fn expired_write_token_is_unauthorized() {
        let key = flakecache_crypto::signing_key_from_seed(&[12; 32]);
        let (addr, _dir) = spawn_auth_server(key.verifying_key());
        let token = token(&key, "write", 1);
        assert_eq!(
            request_with_auth(addr, "PUT", "/expired", b"data", Some(&token)).0,
            401
        );
    }

    #[test]
    fn bad_signature_is_unauthorized() {
        let trusted = flakecache_crypto::signing_key_from_seed(&[13; 32]);
        let untrusted = flakecache_crypto::signing_key_from_seed(&[14; 32]);
        let (addr, _dir) = spawn_auth_server(trusted.verifying_key());
        let token = token(&untrusted, "write", future_exp());
        assert_eq!(
            request_with_auth(addr, "PUT", "/bad-signature", b"data", Some(&token)).0,
            401
        );
    }

    #[test]
    fn read_scope_on_put_is_forbidden() {
        let key = flakecache_crypto::signing_key_from_seed(&[15; 32]);
        let (addr, _dir) = spawn_auth_server(key.verifying_key());
        let token = token(&key, "read", future_exp());
        assert_eq!(
            request_with_auth(addr, "PUT", "/read-only", b"data", Some(&token)).0,
            403
        );
    }

    #[test]
    fn no_auth_key_keeps_put_open() {
        let (addr, _dir) = spawn_server();
        assert_eq!(request(addr, "PUT", "/open", b"data").0, 200);
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

    fn spawn_signed_server() -> (
        std::net::SocketAddr,
        flakecache_crypto::VerifyingKey,
        tempfile::TempDir,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let cas = Cas::new(FilesystemBackend::new(dir.path().join("cas")));
        let meta = MetaStore::open(dir.path().join("meta.redb")).unwrap();
        let node = Node::new(cas, meta);
        let key = flakecache_crypto::signing_key_from_seed(&[7u8; 32]);
        let pubkey = key.verifying_key();
        let server = Arc::new(NixCacheServer::new(node).with_signing_key("testcache-1", key));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let _ = thread::spawn(move || server.serve(&listener));
        (addr, pubkey, dir)
    }

    #[test]
    fn served_narinfo_is_signed_and_verifies() {
        let (addr, pubkey, _dir) = spawn_signed_server();
        // A complete but unsigned narinfo, as a stock `nix copy` would upload.
        let uploaded = "StorePath: /nix/store/00000000000000000000000000000000-x\n\
             URL: nar/abc.nar\n\
             Compression: none\n\
             NarHash: sha256:0984x2ah27wvdh510a7jyqgz2760z20gl44xxrcli8bfnv39ny9c\n\
             NarSize: 5\n\
             References: \n";
        let path = "/00000000000000000000000000000000.narinfo";
        let (put, _) = request(addr, "PUT", path, uploaded.as_bytes());
        assert_eq!(put, 200);

        let (get, body) = request(addr, "GET", path, b"");
        assert_eq!(get, 200);
        let text = String::from_utf8(body).unwrap();
        assert!(
            text.contains("Sig: testcache-1:"),
            "served narinfo unsigned: {text}"
        );

        // The served signature verifies against the cache's public key.
        let (ni, sigs) = narinfo::NarInfo::parse(&text, "/nix/store").unwrap();
        let trusted = std::collections::BTreeMap::from([("testcache-1".to_string(), pubkey)]);
        assert!(
            ni.verify_any(&sigs, &trusted),
            "served signature must verify: {text}"
        );
    }

    #[test]
    fn webdav_verbs_support_an_sccache_backend() {
        let (addr, _dir) = spawn_server();
        // PROPFIND returns a 207 multistatus shaped like the one opendal's
        // WebDAV client parses (else its write-check marks storage read-only).
        let (code, body) = request(addr, "PROPFIND", "/default/sccache/p/", b"");
        assert_eq!(code, 207);
        let text = String::from_utf8(body).unwrap();
        assert!(text.contains("<D:multistatus"), "got: {text}");
        assert!(text.contains("<D:resourcetype><D:collection/></D:resourcetype>"));
        assert!(text.contains("<D:status>HTTP/1.1 200 OK</D:status>"));
        // MKCOL / OPTIONS succeed so directory setup doesn't fail the client.
        assert_eq!(request(addr, "MKCOL", "/default/sccache/p/", b"").0, 201);
        assert_eq!(request(addr, "OPTIONS", "/default/sccache/p/", b"").0, 200);
        // Blob PUT/GET/DELETE round-trip (the actual cache objects).
        assert_eq!(
            request(addr, "PUT", "/default/sccache/p/k", b"artifact").0,
            200
        );
        let (get_code, got) = request(addr, "GET", "/default/sccache/p/k", b"");
        assert_eq!(get_code, 200);
        assert_eq!(got, b"artifact");
        assert_eq!(request(addr, "DELETE", "/default/sccache/p/k", b"").0, 204);
    }
}
