# flakecache-swarm

Swarm coordination for the FlakeCache data plane. **No Raft** — placement is a
deterministic hash, membership is gossip, convergence is CRDT/anti-entropy.

This crate starts with **placement**:

- **`Placement`** — rendezvous (Highest-Random-Weight) hashing. `owners(chunk, R)`
  returns the `R` nodes that own a chunk, highest-weight first, computed
  identically on every node (no coordinator). Adding/removing a node remaps only
  ~1/N of the keyspace.
- Weights use SHAKE-256 (a stable, we-own-it hash) so every node — regardless of
  Rust version or platform — agrees on ownership.

To follow (same crate): SWIM gossip **membership**, Merkle **anti-entropy** for
hot-replica repair, and a **router** that resolves `owners()` to peer fetches.
Placement + membership together are the leaderless, Garage-style swarm.
