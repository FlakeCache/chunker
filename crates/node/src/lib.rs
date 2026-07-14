// SPDX-License-Identifier: MIT
//! The `FlakeCache` data-plane node assembly.
//!
//! Ties the core crates into a working content-addressed, deduplicating store,
//! with zero external dependencies (only `chunker`, `flakecache-cas`,
//! `flakecache-meta`, and std).
//!
//! [`Node::put`] FastCDC-chunks the input, stores each chunk in the CAS
//! (idempotent, so identical chunks are stored once), records the manifest as a
//! CAS object, and writes the chunk->manifest->refcount edges into the metadata
//! DAG. [`Node::get`] reassembles the exact original bytes from a manifest,
//! verifying every chunk on read. Two artifacts that share content share chunks,
//! so the unique stored-chunk count is strictly below the chunk-reference count.

use std::error::Error;
use std::fmt;

use flakecache_cas::{BlobBackend, Cas, CasError, ContentId, ObjectKind};
use flakecache_meta::{MetaError, MetaStore};

/// Errors from the node assembly.
#[derive(Debug)]
pub enum NodeError {
    /// The chunker rejected the input.
    Chunk(String),
    /// A content-addressed store operation failed.
    Cas(CasError),
    /// A metadata operation failed.
    Meta(MetaError),
    /// A manifest referenced a chunk absent from the store (corruption or a GC race).
    MissingChunk(String),
}

impl fmt::Display for NodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Chunk(msg) => write!(f, "chunking: {msg}"),
            Self::Cas(err) => write!(f, "cas: {err}"),
            Self::Meta(err) => write!(f, "meta: {err}"),
            Self::MissingChunk(id) => write!(f, "missing chunk {id} while reassembling"),
        }
    }
}

impl Error for NodeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Cas(err) => Some(err),
            Self::Meta(err) => Some(err),
            Self::Chunk(_) | Self::MissingChunk(_) => None,
        }
    }
}

impl From<CasError> for NodeError {
    fn from(err: CasError) -> Self {
        Self::Cas(err)
    }
}

impl From<MetaError> for NodeError {
    fn from(err: MetaError) -> Self {
        Self::Meta(err)
    }
}

/// A data-plane node: a content-addressed store ([`Cas`]) plus a metadata DAG
/// ([`MetaStore`]) with `FastCDC` chunking and dedup.
#[derive(Debug)]
pub struct Node<B: BlobBackend> {
    cas: Cas<B>,
    meta: MetaStore,
    min: Option<usize>,
    avg: Option<usize>,
    max: Option<usize>,
}

impl<B: BlobBackend> Node<B> {
    /// Assemble a node from a content store and a metadata store (default `FastCDC` sizes).
    pub const fn new(cas: Cas<B>, meta: MetaStore) -> Self {
        Self {
            cas,
            meta,
            min: None,
            avg: None,
            max: None,
        }
    }

    /// Override the `FastCDC` min/avg/max chunk sizes.
    #[must_use]
    pub const fn with_chunk_sizes(mut self, min: usize, avg: usize, max: usize) -> Self {
        self.min = Some(min);
        self.avg = Some(avg);
        self.max = Some(max);
        self
    }

    /// Borrow the content-addressed store.
    pub const fn cas(&self) -> &Cas<B> {
        &self.cas
    }

    /// Borrow the metadata store.
    pub const fn meta(&self) -> &MetaStore {
        &self.meta
    }

    /// Ingest a byte stream: chunk it, store each chunk (deduplicated), record
    /// the manifest, and return the manifest's content id.
    ///
    /// # Errors
    /// Returns a [`NodeError`] if chunking, a store write, or the DAG update fails.
    pub fn put(&self, bytes: &[u8]) -> Result<ContentId, NodeError> {
        let chunks = chunker::chunking::chunk_data(bytes, self.min, self.avg, self.max)
            .map_err(|err| NodeError::Chunk(err.to_string()))?;

        let mut chunk_ids = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            chunk_ids.push(self.cas.put(ObjectKind::Chunk, &chunk.payload)?);
        }

        // The manifest is itself a content-addressed object (the ordered chunk
        // ids), so a peer can fetch it by id; the DAG records it for GC.
        let mut manifest_bytes = Vec::with_capacity(chunk_ids.len() * 32);
        for id in &chunk_ids {
            manifest_bytes.extend_from_slice(id.as_bytes());
        }
        let manifest_id = self.cas.put(ObjectKind::Level1Manifest, &manifest_bytes)?;
        self.meta.put_manifest(manifest_id, &chunk_ids)?;
        Ok(manifest_id)
    }

    /// Reassemble the exact bytes of the artifact addressed by `manifest`, or
    /// `None` if the manifest is unknown.
    ///
    /// # Errors
    /// Returns [`NodeError::MissingChunk`] if a referenced chunk is absent, or a
    /// store error on read/verification failure.
    pub fn get(&self, manifest: ContentId) -> Result<Option<Vec<u8>>, NodeError> {
        let Some(chunk_ids) = self.meta.get_manifest(manifest)? else {
            return Ok(None);
        };
        let mut out = Vec::new();
        for id in chunk_ids {
            let bytes = self
                .cas
                .get(ObjectKind::Chunk, id)?
                .ok_or_else(|| NodeError::MissingChunk(id.to_hex()))?;
            out.extend_from_slice(&bytes);
        }
        Ok(Some(out))
    }

    /// Anchor a GC root: `tag` keeps `manifest` (and its chunks) alive.
    ///
    /// # Errors
    /// Returns a [`NodeError`] if the metadata write fails.
    pub fn set_tag(&self, tag: &str, manifest: ContentId) -> Result<(), NodeError> {
        self.meta.set_root(tag, manifest)?;
        Ok(())
    }

    /// The manifest anchored by `tag`, or `None` if the tag is unset.
    ///
    /// The inverse of [`Node::set_tag`]: a durable name -> manifest lookup, so a
    /// caller that ingested bytes under a tag can later retrieve them by that tag
    /// across process restarts.
    ///
    /// # Errors
    /// Returns a [`NodeError`] if the metadata read fails.
    pub fn get_tag(&self, tag: &str) -> Result<Option<ContentId>, NodeError> {
        Ok(self.meta.get_root(tag)?)
    }

    /// The chunks no longer reachable from any tag (the GC sweep set).
    ///
    /// # Errors
    /// Returns a [`NodeError`] if the metadata read fails.
    pub fn collectible_chunks(&self) -> Result<Vec<ContentId>, NodeError> {
        Ok(self.meta.collectible_chunks()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flakecache_backend::MemoryBackend;
    use std::collections::HashSet;

    /// Deterministic pseudo-random bytes (xorshift64) — no `rand` dependency.
    fn pseudo_random(seed: u64, len: usize) -> Vec<u8> {
        let mut state = seed | 1;
        let mut out = Vec::with_capacity(len + 8);
        while out.len() < len {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            out.extend_from_slice(&state.to_le_bytes());
        }
        out.truncate(len);
        out
    }

    fn node() -> (Node<MemoryBackend>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let cas = Cas::new(MemoryBackend::new());
        let meta = MetaStore::open(dir.path().join("meta.redb")).unwrap();
        // Small FastCDC sizes so modest test inputs produce many chunks.
        let node = Node::new(cas, meta).with_chunk_sizes(4096, 16_384, 65_536);
        (node, dir)
    }

    #[test]
    fn put_get_round_trip_is_byte_exact() -> Result<(), NodeError> {
        let (node, _dir) = node();
        let data = pseudo_random(42, 500_000);
        let manifest = node.put(&data)?;
        assert_eq!(node.get(manifest)?.as_deref(), Some(&data[..]));
        assert_eq!(
            node.get(ContentId::compute(ObjectKind::Level1Manifest, b"nope"))?,
            None
        );
        Ok(())
    }

    #[test]
    fn identical_input_is_fully_deduplicated() -> Result<(), NodeError> {
        let (node, _dir) = node();
        let data = pseudo_random(7, 400_000);
        let m1 = node.put(&data)?;
        let stored_after_first = node.cas().backend().len();
        let m2 = node.put(&data)?; // same bytes
        assert_eq!(m1, m2, "identical input yields the same manifest id");
        assert_eq!(
            node.cas().backend().len(),
            stored_after_first,
            "re-ingesting identical data stores no new objects"
        );
        Ok(())
    }

    #[test]
    fn shared_content_shares_chunks() -> Result<(), NodeError> {
        let (node, _dir) = node();
        let base = pseudo_random(1, 400_000);
        let mut extended = base.clone();
        extended.extend_from_slice(&pseudo_random(2, 150_000)); // base ++ new tail

        let m1 = node.put(&base)?;
        let m2 = node.put(&extended)?;

        let ids1 = node.meta().get_manifest(m1)?.unwrap();
        let ids2 = node.meta().get_manifest(m2)?.unwrap();
        let refs = ids1.len() + ids2.len();
        let unique: HashSet<&ContentId> = ids1.iter().chain(&ids2).collect();

        assert!(
            unique.len() < refs,
            "dedup: {} unique chunks < {} references",
            unique.len(),
            refs
        );
        // Both artifacts still reassemble exactly.
        assert_eq!(node.get(m1)?.as_deref(), Some(&base[..]));
        assert_eq!(node.get(m2)?.as_deref(), Some(&extended[..]));
        Ok(())
    }

    #[test]
    fn tag_round_trips_to_manifest() -> Result<(), NodeError> {
        let (node, _dir) = node();
        let data = pseudo_random(11, 120_000);
        let manifest = node.put(&data)?;

        assert_eq!(node.get_tag("unset")?, None);
        node.set_tag("release", manifest)?;
        assert_eq!(
            node.get_tag("release")?,
            Some(manifest),
            "tag resolves to the manifest"
        );
        // And the tag is a durable retrieval handle: resolve then reassemble.
        let resolved = node.get_tag("release")?.unwrap();
        assert_eq!(node.get(resolved)?.as_deref(), Some(&data[..]));
        Ok(())
    }

    #[test]
    fn gc_identifies_unreferenced_chunks() -> Result<(), NodeError> {
        let (node, _dir) = node();
        let data = pseudo_random(9, 200_000);
        let manifest = node.put(&data)?;

        // No tag -> its chunks are collectible.
        assert!(!node.collectible_chunks()?.is_empty());
        // Tag it -> nothing collectible.
        node.set_tag("release", manifest)?;
        assert!(
            node.collectible_chunks()?.is_empty(),
            "tagged artifact keeps its chunks"
        );
        Ok(())
    }
}
