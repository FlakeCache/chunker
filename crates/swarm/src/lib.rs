// SPDX-License-Identifier: MIT
//! Swarm coordination for the `FlakeCache` data plane.
//!
//! No Raft: placement is a deterministic hash, so every node independently agrees
//! on which nodes own a chunk without a coordinator. This crate provides the
//! placement layer; SWIM gossip membership and Merkle anti-entropy follow.
//!
//! [`Placement`] implements rendezvous (Highest-Random-Weight) hashing: for a
//! key and a node set, each node's weight is a stable hash of `(node, key)` and
//! the key's owners are the highest-weighted nodes. Adding or removing a node
//! remaps only ~1/N of the keyspace, and — because weights use SHAKE-256 rather
//! than a platform hash — every node computes the same ownership regardless of
//! Rust version or platform.

use flakecache_cas::ContentId;

mod config;
mod policy;
pub mod router;
pub mod transport;
pub use config::{FabricConfig, FabricConfigError};
pub use policy::{
    BackendKind, PlacementPlan, PlacementPolicy, PolicyError, ServiceObjective, StorageResource,
};
pub use router::{PeerClient, PeerError, Router, RouterError};
pub use transport::{DEFAULT_TIMEOUT, HttpPeerClient};

/// A node identity in the swarm (e.g. a hostname or stable node id).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(String);

impl NodeId {
    /// Create a node id.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The id as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for NodeId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

/// The rendezvous weight of `key` for `node`: the leading 64 bits of
/// `SHAKE256(node || key)`, big weight = strong claim.
fn weight(node: &NodeId, key: ContentId) -> u64 {
    let mut input = Vec::with_capacity(node.0.len() + 32);
    input.extend_from_slice(node.0.as_bytes());
    input.extend_from_slice(key.as_bytes());
    let digest = flakecache_crypto::shake256_256(&input);
    let mut lead = [0_u8; 8];
    lead.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(lead)
}

/// A rendezvous-hash placement over a set of swarm nodes.
///
/// The node set is kept sorted and de-duplicated so placement is a pure function
/// of the *set* of members, independent of insertion order.
#[derive(Debug, Clone, Default)]
pub struct Placement {
    nodes: Vec<NodeId>,
}

impl Placement {
    /// Build a placement from a set of nodes (de-duplicated and sorted).
    pub fn new(nodes: impl IntoIterator<Item = NodeId>) -> Self {
        let mut nodes: Vec<NodeId> = nodes.into_iter().collect();
        nodes.sort();
        nodes.dedup();
        Self { nodes }
    }

    /// The current members, sorted.
    #[must_use]
    pub fn nodes(&self) -> &[NodeId] {
        &self.nodes
    }

    /// The number of members.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the placement has no members.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Add a member (no-op if already present).
    pub fn add_node(&mut self, node: NodeId) {
        if let Err(pos) = self.nodes.binary_search(&node) {
            self.nodes.insert(pos, node);
        }
    }

    /// Remove a member (no-op if absent).
    pub fn remove_node(&mut self, node: &NodeId) {
        if let Ok(pos) = self.nodes.binary_search(node) {
            self.nodes.remove(pos);
        }
    }

    /// The `replicas` owners of `key`, highest-weight first.
    ///
    /// Returns at most `min(replicas, len())` distinct nodes. Ties (equal
    /// weight) are broken by node id so the result is fully deterministic.
    #[must_use]
    pub fn owners(&self, key: ContentId, replicas: usize) -> Vec<&NodeId> {
        let mut weighted: Vec<(u64, &NodeId)> = self
            .nodes
            .iter()
            .map(|node| (weight(node, key), node))
            .collect();
        weighted.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));
        weighted
            .into_iter()
            .take(replicas)
            .map(|(_, node)| node)
            .collect()
    }

    /// The primary (highest-weight) owner of `key`, if any members exist.
    #[must_use]
    pub fn primary(&self, key: ContentId) -> Option<&NodeId> {
        self.owners(key, 1).into_iter().next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flakecache_cas::ObjectKind;

    fn key(seed: &[u8]) -> ContentId {
        ContentId::compute(ObjectKind::Chunk, seed)
    }

    fn nodes(names: &[&str]) -> Vec<NodeId> {
        names.iter().map(|n| NodeId::new(*n)).collect()
    }

    #[test]
    fn placement_is_insertion_order_independent() {
        let a = Placement::new(nodes(&["n1", "n2", "n3", "n4"]));
        let b = Placement::new(nodes(&["n4", "n2", "n1", "n3"]));
        let k = key(b"chunk");
        assert_eq!(a.owners(k, 3), b.owners(k, 3));
        assert_eq!(a.primary(k), b.primary(k));
    }

    #[test]
    fn owners_are_distinct_and_capped() {
        let p = Placement::new(nodes(&["n1", "n2", "n3"]));
        let k = key(b"x");
        let owners = p.owners(k, 2);
        assert_eq!(owners.len(), 2);
        assert_ne!(owners[0], owners[1]);
        // Asking for more replicas than nodes yields all nodes, once each.
        assert_eq!(p.owners(k, 10).len(), 3);
        assert!(Placement::default().primary(k).is_none());
    }

    #[test]
    fn removing_a_non_owner_does_not_move_the_key() {
        let mut p = Placement::new(nodes(&["n1", "n2", "n3", "n4", "n5"]));
        let k = key(b"stable");
        let before = p.primary(k).unwrap().clone();
        // Remove some node that is NOT the primary owner of this key.
        let victim = p.nodes().iter().find(|n| **n != before).unwrap().clone();
        p.remove_node(&victim);
        assert_eq!(
            p.primary(k),
            Some(&before),
            "unrelated removal must not remap"
        );
    }

    #[test]
    fn adding_a_node_remaps_only_a_minority_of_keys() {
        let mut p = Placement::new(nodes(&["n1", "n2", "n3", "n4"]));
        let keys: Vec<ContentId> = (0..1000_u32).map(|i| key(&i.to_le_bytes())).collect();
        let before: Vec<NodeId> = keys
            .iter()
            .map(|k| p.primary(*k).unwrap().clone())
            .collect();

        p.add_node(NodeId::new("n5"));
        let moved = keys
            .iter()
            .zip(&before)
            .filter(|(k, was)| p.primary(**k).unwrap() != *was)
            .count();

        // Ideal rendezvous remap on 4 -> 5 nodes is ~1/5 of keys; allow slack.
        assert!(
            moved < 350,
            "adding a node remapped {moved}/1000 keys (expected ~200)"
        );
    }

    #[test]
    fn distribution_is_roughly_balanced() {
        let p = Placement::new(nodes(&["n1", "n2", "n3", "n4"]));
        let mut counts = std::collections::HashMap::new();
        for i in 0..4000_u32 {
            let owner = p.primary(key(&i.to_le_bytes())).unwrap().clone();
            *counts.entry(owner).or_insert(0_u32) += 1;
        }
        // Each of 4 nodes should own a healthy share of 4000 keys (ideal 1000).
        for (node, count) in &counts {
            assert!(
                (500..1500).contains(count),
                "node {} owns {count} keys (expected ~1000)",
                node.as_str()
            );
        }
        assert_eq!(counts.len(), 4, "every node owns some keys");
    }
}
