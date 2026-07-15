// SPDX-License-Identifier: MIT
//! Content routing: resolve a chunk's owners and fetch it, local-first.
//!
//! [`Router`] ties [`crate::Placement`] to actual fetching. `get` serves from
//! this node's local backend when present, otherwise fetches from an owner peer
//! (via a [`PeerClient`]) and promotes the bytes into the local backend so the
//! next read is local — the same hot-spreading behaviour as an image mirror.
//! The transport is abstracted behind [`PeerClient`]; a real HTTP/QUIC client is
//! a follow-up, and tests use an in-memory stand-in.

use std::io;

use bytes::Bytes;
use flakecache_cas::{BlobBackend, CasError, ContentId};

use crate::{NodeId, Placement};

/// A transport-agnostic error from a peer operation.
pub type PeerError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Transfer blobs to and from a specific peer node.
pub trait PeerClient {
    /// Fetch `id` from `node`, or `None` if that peer does not have it.
    ///
    /// # Errors
    /// Returns a [`PeerError`] if the peer transport fails.
    fn fetch(&self, node: &NodeId, id: ContentId) -> Result<Option<Bytes>, PeerError>;

    /// Replicate `bytes` (a blob with id `id`) onto `node`. Idempotent, since
    /// content is immutable and content-addressed.
    ///
    /// # Errors
    /// Returns a [`PeerError`] if the peer transport fails. Callers replicating
    /// on write treat this as best-effort and do not fail the local write.
    fn push(&self, node: &NodeId, id: ContentId, bytes: &[u8]) -> Result<(), PeerError>;
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
        self.read_local_first(id)
    }

    /// The local-first-then-owner-peer read shared by the inherent [`Self::get`]
    /// and the [`BlobBackend`] implementation.
    fn read_local_first(&self, id: ContentId) -> Result<Option<Bytes>, RouterError> {
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

impl<Local: BlobBackend + Sync, Peers: PeerClient + Sync> Router<Local, Peers> {
    /// The co-owner peers of `id`: its owners minus this node. This node keeps
    /// its own copy locally, so replication targets only the *other* owners
    /// (whether or not this node is itself an owner of `id`).
    fn replica_peers(&self, id: ContentId) -> Vec<&NodeId> {
        self.placement
            .owners(id, self.replicas)
            .into_iter()
            .filter(|owner| **owner != self.me)
            .collect()
    }

    /// Push `bytes` to every co-owner peer of `id` in parallel, best-effort.
    ///
    /// Each push runs on its own scoped thread with the peer client's per-request
    /// timeout, so a slow or dead peer bounds only its own thread and never the
    /// sum of all peers. A push failure is logged and dropped: the local write
    /// has already succeeded, and anti-entropy (a later increment) reconciles a
    /// replica that a transient failure left behind.
    fn replicate(&self, id: ContentId, bytes: &[u8]) {
        let peers = self.replica_peers(id);
        if peers.is_empty() {
            return;
        }
        std::thread::scope(|scope| {
            for peer in peers {
                scope.spawn(move || {
                    if let Err(error) = self.peers.push(peer, id, bytes) {
                        eprintln!(
                            "flakecache-swarm: replicate {} to {} failed: {error}",
                            id.to_hex(),
                            peer.as_str(),
                        );
                    }
                });
            }
        });
    }
}

/// [`Router`] is a drop-in [`BlobBackend`]: `put` writes locally and then
/// replicates to co-owner peers on the write path, `get` reads local-first then
/// from owner peers. This lets a node's server front-end store and serve chunks
/// through the router without knowing the swarm exists.
impl<Local: BlobBackend + Sync, Peers: PeerClient + Sync> BlobBackend for Router<Local, Peers> {
    fn put(&self, id: ContentId, bytes: &[u8]) -> Result<(), CasError> {
        // Durable locally first; only then fan out replicas. A replica push that
        // fails must not fail the write, so replication is best-effort.
        self.local.put(id, bytes)?;
        self.replicate(id, bytes);
        Ok(())
    }

    fn get(&self, id: ContentId) -> Result<Option<Bytes>, CasError> {
        // Reuse the local-first-then-peer read path, mapping the router error
        // onto the backend's error: a local failure is preserved verbatim, a
        // peer transport failure surfaces as I/O.
        self.read_local_first(id).map_err(|error| match error {
            RouterError::Local(cas) => cas,
            RouterError::Peer(peer) => CasError::Io(io::Error::other(peer)),
        })
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
            let mut stores = self
                .stores
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let bytes = match stores.get(node) {
                Some(store) => store.get(cid).map_err(|e| Box::new(e) as PeerError)?,
                None => None,
            };
            if bytes.is_some() {
                stores.remove(node);
            }
            Ok(bytes)
        }

        fn push(&self, node: &NodeId, cid: ContentId, bytes: &[u8]) -> Result<(), PeerError> {
            let mut stores = self
                .stores
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            stores
                .entry(node.clone())
                .or_default()
                .put(cid, bytes)
                .map_err(|e| Box::new(e) as PeerError)
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
        owner_store
            .put(cid, b"remote")
            .map_err(RouterError::Local)?;
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
        let router = Router::new(
            NodeId::new("n1"),
            placement(),
            MemoryBackend::new(),
            empty_peers(),
            3,
        );
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
        let outsider = Router::new(
            NodeId::new("zzz"),
            placement(),
            MemoryBackend::new(),
            empty_peers(),
            2,
        );
        assert!(!outsider.is_owner(cid));
    }

    /// A peer client that records every `push` it receives, so a test can assert
    /// exactly which co-owners a write replicated to. `fetch` is unused here.
    #[derive(Default)]
    struct RecordingPeers {
        pushes: Mutex<Vec<(NodeId, ContentId, Vec<u8>)>>,
    }

    impl RecordingPeers {
        fn targets(&self) -> Vec<NodeId> {
            let mut nodes: Vec<NodeId> = self
                .pushes
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .iter()
                .map(|(node, _, _)| node.clone())
                .collect();
            nodes.sort();
            nodes
        }
    }

    impl PeerClient for RecordingPeers {
        fn fetch(&self, _node: &NodeId, _cid: ContentId) -> Result<Option<Bytes>, PeerError> {
            Ok(None)
        }

        fn push(&self, node: &NodeId, cid: ContentId, bytes: &[u8]) -> Result<(), PeerError> {
            self.pushes
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push((node.clone(), cid, bytes.to_vec()));
            Ok(())
        }
    }

    // `Router::put` (via BlobBackend) writes locally, then replicates to every
    // co-owner peer in parallel — the owners of the key minus this node.
    #[test]
    fn put_writes_local_and_replicates_to_co_owners() -> Result<(), CasError> {
        let cid = id(b"replicated");
        let p = placement();
        let owners: Vec<NodeId> = p.owners(cid, 3).into_iter().cloned().collect();
        // Run as the primary owner: replicas are the other two owners, not self.
        let me = owners[0].clone();
        let expected: Vec<NodeId> = {
            let mut rest = owners[1..].to_vec();
            rest.sort();
            rest
        };

        let local = MemoryBackend::new();
        let peers = RecordingPeers::default();
        let router = Router::new(me, p, local, peers, 3);

        BlobBackend::put(&router, cid, b"replicated")?;

        // The write is durable locally, and served through the router's get.
        assert_eq!(
            BlobBackend::get(&router, cid)?.as_deref(),
            Some(&b"replicated"[..]),
        );
        assert_eq!(
            router.peers.targets(),
            expected,
            "replicated to exactly the co-owners, never to self",
        );
        assert!(
            !router.peers.targets().contains(&router.me),
            "never replicate to self",
        );
        Ok(())
    }

    // A non-owner node that ingests a chunk keeps a local copy and pushes to all
    // owners (the full owner set, since none of them is self).
    #[test]
    fn non_owner_put_replicates_to_all_owners() -> Result<(), CasError> {
        let cid = id(b"chunk");
        let p = placement();
        let mut expected: Vec<NodeId> = p.owners(cid, 2).into_iter().cloned().collect();
        expected.sort();

        // "zzz" is not a member, so it is never an owner of `cid`.
        let router = Router::new(
            NodeId::new("zzz"),
            p,
            MemoryBackend::new(),
            RecordingPeers::default(),
            2,
        );
        BlobBackend::put(&router, cid, b"chunk")?;
        assert_eq!(router.peers.targets(), expected);
        Ok(())
    }

    /// A peer client whose `push` always fails, to prove a replica failure is
    /// logged but never fails the local write.
    struct FailingPeers;

    impl PeerClient for FailingPeers {
        fn fetch(&self, _node: &NodeId, _cid: ContentId) -> Result<Option<Bytes>, PeerError> {
            Ok(None)
        }

        fn push(&self, _node: &NodeId, _cid: ContentId, _bytes: &[u8]) -> Result<(), PeerError> {
            Err(Box::<dyn std::error::Error + Send + Sync>::from("peer down"))
        }
    }

    #[test]
    fn replica_failure_does_not_fail_the_write() -> Result<(), CasError> {
        let cid = id(b"durable");
        let local = MemoryBackend::new();
        let router = Router::new(NodeId::new("n1"), placement(), local, FailingPeers, 3);

        // Every replica push errors, yet the local write succeeds and is served.
        BlobBackend::put(&router, cid, b"durable")?;
        assert_eq!(
            BlobBackend::get(&router, cid)?.as_deref(),
            Some(&b"durable"[..]),
        );
        Ok(())
    }
}
