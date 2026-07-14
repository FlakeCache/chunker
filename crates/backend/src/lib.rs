// SPDX-License-Identifier: MIT
//! Blob backends for `FlakeCache`.
//!
//! Implementations of the CAS [`BlobBackend`] trait beyond the local warm-disk
//! backend in `flakecache-cas`:
//!
//! - [`MemoryBackend`] — an in-process thread-safe store (tests, ephemeral use).
//! - [`S3Backend`] — a durable, path-style, AWS-SigV4 S3-compatible cold tier.
//! - [`TieredBackend`] — warm/cold tiering: `put` writes the durable cold tier
//!   first then warm; `get` serves from warm, falling back to cold and promoting
//!   (write-back) the bytes into warm so the next read is local.
//!
//! Nothing here stores a whole undeduplicated blob; callers store `FastCDC`
//! chunks addressed by content hash.

use std::collections::HashMap;
use std::sync::{PoisonError, RwLock};

use bytes::Bytes;
use flakecache_cas::{BlobBackend, CasError, ContentId};

mod s3;

pub use s3::{S3Backend, S3Config};

/// An in-process, thread-safe blob store backed by a `HashMap`.
///
/// A poisoned lock (from a panic elsewhere while holding it) is recovered rather
/// than propagated, since the stored map is not left in an inconsistent state by
/// a panicking reader/writer.
#[derive(Debug, Default)]
pub struct MemoryBackend {
    blobs: RwLock<HashMap<[u8; 32], Bytes>>,
}

impl MemoryBackend {
    /// Create an empty backend.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of stored blobs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.blobs
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Whether the backend holds no blobs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether a blob is stored under `id`.
    #[must_use]
    pub fn contains(&self, id: ContentId) -> bool {
        self.blobs
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .contains_key(id.as_bytes())
    }
}

impl BlobBackend for MemoryBackend {
    fn put(&self, id: ContentId, bytes: &[u8]) -> Result<(), CasError> {
        self.blobs
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(*id.as_bytes())
            .or_insert_with(|| Bytes::copy_from_slice(bytes));
        Ok(())
    }

    fn get(&self, id: ContentId) -> Result<Option<Bytes>, CasError> {
        Ok(self
            .blobs
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(id.as_bytes())
            .cloned())
    }
}

/// Hot/cold tiering over a warm and a cold [`BlobBackend`].
///
/// The cold tier is the durable system of record; the warm tier is a fast local
/// cache. Durability is achieved on `put` before it returns.
#[derive(Debug, Clone)]
pub struct TieredBackend<Warm, Cold> {
    warm: Warm,
    cold: Cold,
}

impl<Warm, Cold> TieredBackend<Warm, Cold> {
    /// Compose a `warm` and a `cold` backend.
    pub const fn new(warm: Warm, cold: Cold) -> Self {
        Self { warm, cold }
    }

    /// Borrow the warm tier.
    pub const fn warm(&self) -> &Warm {
        &self.warm
    }

    /// Borrow the cold tier.
    pub const fn cold(&self) -> &Cold {
        &self.cold
    }
}

impl<Warm: BlobBackend, Cold: BlobBackend> BlobBackend for TieredBackend<Warm, Cold> {
    fn put(&self, id: ContentId, bytes: &[u8]) -> Result<(), CasError> {
        // Durability first: write the cold system-of-record, then warm. A put
        // that returns Ok is durable even if the warm disk is later lost.
        self.cold.put(id, bytes)?;
        self.warm.put(id, bytes)?;
        Ok(())
    }

    fn get(&self, id: ContentId) -> Result<Option<Bytes>, CasError> {
        if let Some(bytes) = self.warm.get(id)? {
            return Ok(Some(bytes));
        }
        let Some(bytes) = self.cold.get(id)? else {
            return Ok(None);
        };
        // Promote (write-back) into warm so the next read is local. Best-effort:
        // the bytes are already in hand, so a warm-write failure is not fatal.
        let _ = self.warm.put(id, &bytes);
        Ok(Some(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flakecache_cas::{FilesystemBackend, ObjectKind};

    fn id(seed: &[u8]) -> ContentId {
        ContentId::compute(ObjectKind::Chunk, seed)
    }

    #[test]
    fn memory_backend_put_get_and_dedup() -> Result<(), CasError> {
        let mem = MemoryBackend::new();
        assert!(mem.is_empty());
        let i = id(b"data");
        mem.put(i, b"data")?;
        mem.put(i, b"data")?; // idempotent
        assert_eq!(mem.len(), 1);
        assert!(mem.contains(i));
        assert_eq!(mem.get(i)?.as_deref(), Some(&b"data"[..]));
        assert!(mem.get(id(b"absent"))?.is_none());
        Ok(())
    }

    #[test]
    fn tiered_put_writes_through_to_both() -> Result<(), CasError> {
        let tier = TieredBackend::new(MemoryBackend::new(), MemoryBackend::new());
        let i = id(b"x");
        tier.put(i, b"x")?;
        assert!(tier.warm().contains(i), "warm has it");
        assert!(tier.cold().contains(i), "cold (durable) has it");
        assert_eq!(tier.get(i)?.as_deref(), Some(&b"x"[..]));
        Ok(())
    }

    #[test]
    fn tiered_cold_hit_promotes_into_warm() -> Result<(), CasError> {
        let warm = MemoryBackend::new();
        let cold = MemoryBackend::new();
        let i = id(b"y");
        cold.put(i, b"y")?; // present only in cold (warm cold-started / evicted)
        let tier = TieredBackend::new(warm, cold);

        assert!(!tier.warm().contains(i), "not warm yet");
        assert_eq!(tier.get(i)?.as_deref(), Some(&b"y"[..]), "served from cold");
        assert!(tier.warm().contains(i), "cold hit promotes into warm");
        Ok(())
    }

    #[test]
    fn tiered_miss_is_none() -> Result<(), CasError> {
        let tier = TieredBackend::new(MemoryBackend::new(), MemoryBackend::new());
        assert!(tier.get(id(b"absent"))?.is_none());
        Ok(())
    }

    #[test]
    fn tiered_over_local_disk_warm() -> Result<(), CasError> {
        // Realistic shape: local disk warm tier, in-memory stand-in for cold.
        let dir = tempfile::tempdir().unwrap();
        let tier = TieredBackend::new(FilesystemBackend::new(dir.path()), MemoryBackend::new());
        let i = id(b"disk");
        tier.put(i, b"disk")?;
        assert!(tier.cold().contains(i));
        assert_eq!(tier.get(i)?.as_deref(), Some(&b"disk"[..]));
        Ok(())
    }
}
