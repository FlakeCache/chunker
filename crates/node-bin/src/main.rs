// SPDX-License-Identifier: MIT
//! Single-node `FlakeCache` Nix binary-cache server.
//!
//! Wires a [`Node`] (filesystem CAS + redb metadata) into the std-only HTTP/1.1
//! front-end from [`flakecache_proto_nix`] and serves it.
//!
//! Configuration (positional args win over environment, environment over
//! defaults):
//! - listen address: arg 1, else `FLAKECACHE_LISTEN`, else `127.0.0.1:8501`.
//! - data directory: arg 2, else `FLAKECACHE_DATA_DIR`, else `./flakecache-data`.
//! - signing key (optional): `FLAKECACHE_SIGNING_KEY` (a Nix `<name>:<base64>`
//!   secret key) or `FLAKECACHE_SIGNING_KEY_FILE` (path to such a key file).
//!   When set, served narinfos are signed so trusting Nix clients substitute.
//! - token verification key (optional): `FLAKECACHE_TOKEN_PUBKEY`, a standard
//!   base64-encoded 32-byte Ed25519 public key. When set, writes require auth.

use std::env;
use std::error::Error;
use std::fs;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;

use flakecache_cas::{Cas, FilesystemBackend};
use flakecache_meta::MetaStore;
use flakecache_node::Node;
use flakecache_proto_nix::{NixCacheServer, narinfo};

const DEFAULT_LISTEN: &str = "127.0.0.1:8501";
const DEFAULT_DATA_DIR: &str = "./flakecache-data";

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);

    let listen = args
        .next()
        .or_else(|| env::var("FLAKECACHE_LISTEN").ok())
        .unwrap_or_else(|| DEFAULT_LISTEN.to_owned());

    let data_dir = args
        .next()
        .or_else(|| env::var("FLAKECACHE_DATA_DIR").ok())
        .map_or_else(|| PathBuf::from(DEFAULT_DATA_DIR), PathBuf::from);

    fs::create_dir_all(&data_dir)?;
    let cas = Cas::new(FilesystemBackend::new(data_dir.join("cas")));
    let meta = MetaStore::open(data_dir.join("meta.redb"))?;
    let node = Node::new(cas, meta);
    let mut server = NixCacheServer::new(node);

    // Optional cache signing key.
    let signing = env::var("FLAKECACHE_SIGNING_KEY").ok().map_or_else(
        || {
            env::var("FLAKECACHE_SIGNING_KEY_FILE")
                .ok()
                .map(fs::read_to_string)
                .transpose()
        },
        |k| Ok(Some(k)),
    )?;
    if let Some(raw) = signing {
        let (name, key) = narinfo::parse_secret_key(&raw)?;
        println!("flakecache-node-bin: signing served narinfos as '{name}'");
        server = server.with_signing_key(name, key);
    }
    if let Ok(raw) = env::var("FLAKECACHE_TOKEN_PUBKEY") {
        let bytes = flakecache_crypto::b64::decode(raw.trim())?;
        let bytes: [u8; 32] = bytes
            .try_into()
            .map_err(|_| "FLAKECACHE_TOKEN_PUBKEY must decode to 32 bytes")?;
        let key = flakecache_crypto::VerifyingKey::from_bytes(&bytes)?;
        println!("flakecache-node-bin: write-token authentication enabled");
        server = server.with_auth_key(key);
    }
    let server = Arc::new(server);

    let listener = TcpListener::bind(&listen)?;
    let addr = listener.local_addr()?;
    println!(
        "flakecache-node-bin: serving Nix binary cache on http://{addr} (data dir: {})",
        data_dir.display()
    );

    server.serve(&listener)?;
    Ok(())
}
