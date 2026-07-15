// SPDX-License-Identifier: MIT
//! Single-node `FlakeCache` Nix binary-cache server.
//!
//! Wires a [`Node`] (filesystem CAS + redb metadata) into the std-only HTTP/1.1
//! front-end from [`flakecache_proto_nix`] and serves it. When S3 configuration
//! is present, the CAS is warm local disk over a durable S3 cold tier.
//!
//! Configuration (positional args win over environment, environment over
//! defaults):
//! - listen address: arg 1, else `FLAKECACHE_LISTEN`, else `127.0.0.1:8501`.
//! - data directory: arg 2, else `FLAKECACHE_DATA_DIR`, else `./flakecache-data`.
//! - cold tier: set the required `FLAKECACHE_S3_*` variables documented by
//!   `flakecache_backend::S3Config::from_env`; absence selects local disk only.
//! - signing key (optional): `FLAKECACHE_SIGNING_KEY` (a Nix `<name>:<base64>`
//!   secret key) or `FLAKECACHE_SIGNING_KEY_FILE` (path to such a key file).
//!   When set, served narinfos are signed so trusting Nix clients substitute.
//! - token verification key (optional): `FLAKECACHE_TOKEN_PUBKEY`, a standard
//!   base64-encoded 32-byte Ed25519 public key. When set, writes require auth.
//! - swarm replication (optional): set `FLAKECACHE_SWARM_SELF` to this node's
//!   peer base URL (e.g. `http://cache-1:8501`) to wrap the CAS in a swarm
//!   [`Router`], replicating each written chunk to its co-owner peers listed in
//!   `FLAKECACHE_SWARM_PEERS` (comma-separated base URLs) with a replica count
//!   from `FLAKECACHE_SWARM_REPLICAS` (default 3). Absent `FLAKECACHE_SWARM_SELF`,
//!   the node behaves exactly as a single-node cache.

use std::env;
use std::error::Error;
use std::fs;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;

use flakecache_backend::{S3Backend, TieredBackend};
use flakecache_cas::{BlobBackend, Cas, FilesystemBackend};
use flakecache_meta::MetaStore;
use flakecache_node::Node;
use flakecache_proto_nix::{NixCacheServer, narinfo};
use flakecache_swarm::{DEFAULT_TIMEOUT, HttpPeerClient, NodeId, Placement, Router};

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
    let listener = TcpListener::bind(&listen)?;
    let addr = listener.local_addr()?;
    println!(
        "flakecache-node-bin: serving Nix binary cache on http://{addr} (data dir: {})",
        data_dir.display()
    );

    let warm = FilesystemBackend::new(data_dir.join("cas"));
    if env::var_os("FLAKECACHE_S3_ENDPOINT").is_some() {
        println!("flakecache-node-bin: S3 cold tier enabled");
        run(
            TieredBackend::new(warm, S3Backend::from_env()?),
            &data_dir,
            signing,
            &listener,
        )?;
    } else {
        run(warm, &data_dir, signing, &listener)?;
    }
    Ok(())
}

/// Serve `backend` directly, or — when swarm replication is configured — wrap it
/// in a [`Router`] first so writes replicate to co-owner peers. When
/// `FLAKECACHE_SWARM_SELF` is unset the router is never constructed and the
/// backend is served exactly as before.
fn run<B: BlobBackend + Send + Sync + 'static>(
    backend: B,
    data_dir: &std::path::Path,
    signing: Option<String>,
    listener: &TcpListener,
) -> Result<(), Box<dyn Error>> {
    match swarm_config()? {
        Some(config) => {
            println!(
                "flakecache-node-bin: swarm replication enabled as '{}' ({} members, {} replicas)",
                config.me.as_str(),
                config.placement.len(),
                config.replicas,
            );
            let router = Router::new(
                config.me,
                config.placement,
                backend,
                HttpPeerClient::new(DEFAULT_TIMEOUT),
                config.replicas,
            );
            serve(router, data_dir, signing, listener)
        }
        None => serve(backend, data_dir, signing, listener),
    }
}

/// Resolved swarm membership for this node.
struct SwarmConfig {
    me: NodeId,
    placement: Placement,
    replicas: usize,
}

/// Default replica count when `FLAKECACHE_SWARM_REPLICAS` is unset.
const DEFAULT_REPLICAS: usize = 3;

/// Build a [`SwarmConfig`] from the environment, or `None` when swarm mode is
/// off (`FLAKECACHE_SWARM_SELF` unset or empty).
///
/// This node's id and every peer id are peer **base URLs** (see
/// [`flakecache_swarm::HttpPeerClient`]); a trailing slash is trimmed so this
/// node's `SELF` form matches the way peers list it. The placement is the peer
/// set plus this node.
fn swarm_config() -> Result<Option<SwarmConfig>, Box<dyn Error>> {
    let Some(me) = env::var("FLAKECACHE_SWARM_SELF")
        .ok()
        .map(|value| value.trim().trim_end_matches('/').to_owned())
        .filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };
    let me = NodeId::new(me);

    let mut members: Vec<NodeId> = env::var("FLAKECACHE_SWARM_PEERS")
        .unwrap_or_default()
        .split(',')
        .map(|peer| peer.trim().trim_end_matches('/'))
        .filter(|peer| !peer.is_empty())
        .map(NodeId::new)
        .collect();
    members.push(me.clone());
    let placement = Placement::new(members);

    let replicas = match env::var("FLAKECACHE_SWARM_REPLICAS") {
        Ok(raw) => raw
            .trim()
            .parse::<usize>()
            .map_err(|e| format!("FLAKECACHE_SWARM_REPLICAS must be a positive integer: {e}"))?
            .max(1),
        Err(_) => DEFAULT_REPLICAS,
    };

    Ok(Some(SwarmConfig {
        me,
        placement,
        replicas,
    }))
}

fn serve<B: BlobBackend + Send + Sync + 'static>(
    backend: B,
    data_dir: &std::path::Path,
    signing: Option<String>,
    listener: &TcpListener,
) -> Result<(), Box<dyn Error>> {
    let meta = MetaStore::open(data_dir.join("meta.redb"))?;
    let node = Node::new(Cas::new(backend), meta);
    let mut server = NixCacheServer::new(node);
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
    server.serve(listener)?;
    Ok(())
}
