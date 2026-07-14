// SPDX-License-Identifier: MIT
//! Content routing: resolve a chunk's owners and fetch it, local-first.
//!
//! [`Router`] ties [`crate::Placement`] to actual fetching. `get` serves from
//! this node's local backend when present, otherwise fetches from an owner peer
//! (via a [`PeerClient`]) and promotes the bytes into the local backend so the
//! next read is local — the same hot-spreading behaviour as an image mirror.
//! The transport is abstracted behind [`PeerClient`]; a real HTTP/QUIC client is
//! a follow-up, and tests use an in-memory stand-in.

use bytes::Bytes;
use flakecache_cas::{BlobBackend, CasError, ContentId};

use crate::{NodeId, Placement};

/// A transport-agnostic error from a peer fetch.
pub type PeerError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Fetch a blob from a specific peer node.
pub trait PeerClient {
    /// Fetch `id` from `node`, or `None` if that peer does not have it.
    ///
    /// # Errors
    /// Returns a [`PeerError`] if the peer transport fails.
    fn fetch(&self, node: &NodeId, id: ContentId) -> Result<Option<Bytes>, PeerError>;
}

/// Errors from the router.
#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    /// The local backend failed.
    #[error("local backend: {0}")]
    Local(#[source] CasError),
    /// A peer fetch failed.
    #[error("peer fetch: {0}")]
    Peer(#[source] PeerError),
}

/// Routes chunk reads across the swarm: local first, then owner peers.
#[derive(Debug)]
pub struct Router<Local, Peers> {
    me: NodeId,
    placement: Placement,
    local: Local,
    peers: Peers,
    replicas: usize,
}

impl<Local: BlobBackend, Peers: PeerClient> Router<Local, Peers> {
    /// Create a router for this node.
    pub const fn new(
        me: NodeId,
        placement: Placement,
        local: Local,
        peers: Peers,
        replicas: usize,
    ) -> Self {
        Self {
            me,
            placement,
            local,
            peers,
            replicas,
        }
    }

    /// The owner nodes for `id` (highest-weight first).
    #[must_use]
    pub fn owners(&self, id: ContentId) -> Vec<&NodeId> {
        self.placement.owners(id, self.replicas)
    }

    /// Whether this node is among the owners of `id` (should hold a replica).
    #[must_use]
    pub fn is_owner(&self, id: ContentId) -> bool {
        self.owners(id).iter().any(|node| **node == self.me)
    }

    /// Store `bytes` in this node's local backend.
    ///
    /// # Errors
    /// Returns [`RouterError::Local`] if the local write fails.
    pub fn put_local(&self, id: ContentId, bytes: &[u8]) -> Result<(), RouterError> {
        self.local.put(id, bytes).map_err(RouterError::Local)
    }

    /// Fetch `id`: from the local backend if present, otherwise from an owner
    /// peer. A peer hit is promoted into the local backend (best-effort cache).
    ///
    /// # Errors
    /// Returns [`RouterError`] if the local read or a peer fetch fails.
    pub fn get(&self, id: ContentId) -> Result<Option<Bytes>, RouterError> {
        if let Some(bytes) = self.local.get(id).map_err(RouterError::Local)? {
            return Ok(Some(bytes));
        }
        for owner in self.placement.owners(id, self.replicas) {
            if *owner == self.me {
                continue; // our own local backend was already checked
            }
            if let Some(bytes) = self.peers.fetch(owner, id).map_err(RouterError::Peer)? {
                // Promote (write-back) so the next read is local; best-effort,
                // since the bytes are already in hand.
                let _ = self.local.put(id, &bytes);
                return Ok(Some(bytes));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flakecache_backend::MemoryBackend;
    use flakecache_cas::ObjectKind;
    use std::collections::HashMap;
    use std::sync::Mutex;

    fn id(seed: &[u8]) -> ContentId {
        ContentId::compute(ObjectKind::Chunk, seed)
    }

    fn placement() -> Placement {
        Placement::new(["n1", "n2", "n3", "n4"].iter().map(|s| NodeId::new(*s)))
    }

    /// A peer client that serves each node's blob at most once, so a second
    /// successful read can only come from a local promotion.
    struct OnceServingPeers {
        stores: Mutex<HashMap<NodeId, MemoryBackend>>,
    }

    impl PeerClient for OnceServingPeers {
        fn fetch(&self, node: &NodeId, cid: ContentId) -> Result<Option<Bytes>, PeerError> {
            let mut stores = self.stores.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let bytes = match stores.get(node) {
                Some(store) => store.get(cid).map_err(|e| Box::new(e) as PeerError)?,
                None => None,
            };
            if bytes.is_some() {
                stores.remove(node);
            }
            Ok(bytes)
        }
    }

    fn empty_peers() -> OnceServingPeers {
        OnceServingPeers {
            stores: Mutex::new(HashMap::new()),
        }
    }

    #[test]
    fn local_hit_short_circuits() -> Result<(), RouterError> {
        let local = MemoryBackend::new();
        let cid = id(b"data");
        local.put(cid, b"data").map_err(RouterError::Local)?;
        let router = Router::new(NodeId::new("n1"), placement(), local, empty_peers(), 3);
        assert_eq!(router.get(cid)?.as_deref(), Some(&b"data"[..]));
        Ok(())
    }

    #[test]
    fn peer_hit_promotes_into_local() -> Result<(), RouterError> {
        let cid = id(b"remote");
        let p = placement();
        let owner = p.owners(cid, 3)[0].clone();

        let owner_store = MemoryBackend::new();
        owner_store.put(cid, b"remote").map_err(RouterError::Local)?;
        let mut stores = HashMap::new();
        stores.insert(owner, owner_store);
        let peers = OnceServingPeers {
            stores: Mutex::new(stores),
        };

        // "zzz" is not a member/owner and starts with an empty local backend.
        let router = Router::new(NodeId::new("zzz"), p, MemoryBackend::new(), peers, 3);

        assert_eq!(
            router.get(cid)?.as_deref(),
            Some(&b"remote"[..]),
            "served from owner peer"
        );
        // The peer served only once; a second hit proves it was promoted local.
        assert_eq!(
            router.get(cid)?.as_deref(),
            Some(&b"remote"[..]),
            "second read must be the promoted local copy"
        );
        Ok(())
    }

    #[test]
    fn miss_everywhere_is_none() -> Result<(), RouterError> {
        let router = Router::new(NodeId::new("n1"), placement(), MemoryBackend::new(), empty_peers(), 3);
        assert!(router.get(id(b"absent"))?.is_none());
        Ok(())
    }

    #[test]
    fn is_owner_reflects_placement() {
        let p = placement();
        let cid = id(b"k");
        let owner_name = p.owners(cid, 2)[0].as_str().to_owned();
        let router = Router::new(
            NodeId::new(owner_name),
            p,
            MemoryBackend::new(),
            empty_peers(),
            2,
        );
        assert!(router.is_owner(cid), "the primary owner reports ownership");
        // A non-member is never an owner.
        let outsider = Router::new(NodeId::new("zzz"), placement(), MemoryBackend::new(), empty_peers(), 2);
        assert!(!outsider.is_owner(cid));
    }
}
