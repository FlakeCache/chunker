// SPDX-License-Identifier: MIT
//! Node-local metadata engine for `FlakeCache`.
//!
//! A disk-resident chunk -> manifest -> refcount DAG with exact mark-and-sweep
//! garbage-collection reachability, backed by [`redb`] (crash-safe, mmap'd,
//! on-disk B-tree — the working set is OS-paged, so the index is not RAM-bound).
//!
//! - [`MetaStore::put_manifest`] records a manifest as an ordered list of chunk
//!   [`ContentId`]s and maintains per-chunk refcounts. It is idempotent, so a
//!   chunk shared by several manifests is stored once and its refcount rises.
//! - [`MetaStore::set_root`] anchors a live reference (a tag) to a manifest.
//! - [`MetaStore::collectible_chunks`] returns tracked chunks NOT reachable from
//!   any live root — the sweep set. The caller deletes those blobs from the CAS.
//!
//! Reachability is the authority for what may be collected; refcounts are an
//! O(1) stat and the basis for future CRDT/PN-counter convergence in the swarm.

use std::collections::BTreeSet;
use std::path::Path;

use flakecache_cas::ContentId;
use redb::{Database, ReadableTable, TableDefinition};

/// manifest id -> concatenated 32-byte chunk ids (ordered).
const MANIFESTS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("manifests");
/// root name (tag) -> manifest id.
const ROOTS: TableDefinition<&str, &[u8]> = TableDefinition::new("roots");
/// chunk id -> reference count.
const REFCOUNT: TableDefinition<&[u8], u64> = TableDefinition::new("refcount");

/// Errors from the metadata engine.
///
/// The redb source is boxed so the error stays small (redb's error enums are
/// large, which would otherwise bloat every `Result` return).
#[derive(Debug, thiserror::Error)]
pub enum MetaError {
    /// An underlying redb (on-disk database) error.
    #[error("redb: {0}")]
    Redb(Box<dyn std::error::Error + Send + Sync + 'static>),
    /// A stored manifest value was not a whole number of 32-byte ids.
    #[error("corrupt manifest: {0} bytes is not a multiple of 32")]
    CorruptManifest(usize),
}

macro_rules! from_redb {
    ($($ty:ty),+ $(,)?) => {
        $(impl From<$ty> for MetaError {
            fn from(err: $ty) -> Self {
                Self::Redb(Box::new(err))
            }
        })+
    };
}
from_redb!(
    redb::DatabaseError,
    redb::TransactionError,
    redb::TableError,
    redb::StorageError,
    redb::CommitError,
);

/// Decode a concatenation of 32-byte content ids.
fn decode_ids(bytes: &[u8]) -> Result<Vec<ContentId>, MetaError> {
    if bytes.len() % 32 != 0 {
        return Err(MetaError::CorruptManifest(bytes.len()));
    }
    let mut ids = Vec::with_capacity(bytes.len() / 32);
    for chunk in bytes.chunks_exact(32) {
        let arr: [u8; 32] = chunk
            .try_into()
            .map_err(|_| MetaError::CorruptManifest(bytes.len()))?;
        ids.push(ContentId::from_bytes(arr));
    }
    Ok(ids)
}

/// The disk-resident metadata store.
#[derive(Debug)]
pub struct MetaStore {
    db: Database,
}

impl MetaStore {
    /// Open (creating if absent) the metadata database at `path`.
    ///
    /// # Errors
    /// Returns [`MetaError::Open`] if the database cannot be created or opened.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, MetaError> {
        Ok(Self {
            db: Database::create(path)?,
        })
    }

    /// Record `manifest` as the ordered list `chunks` and bump each chunk's
    /// refcount. Idempotent: re-inserting an existing manifest id changes
    /// nothing and returns `false`; a newly recorded manifest returns `true`.
    ///
    /// # Errors
    /// Returns a [`MetaError`] if any table operation or the commit fails.
    pub fn put_manifest(
        &self,
        manifest: ContentId,
        chunks: &[ContentId],
    ) -> Result<bool, MetaError> {
        let txn = self.db.begin_write()?;
        let newly_recorded;
        {
            let mut manifests = txn.open_table(MANIFESTS)?;
            if manifests.get(manifest.as_bytes().as_slice())?.is_some() {
                newly_recorded = false;
            } else {
                let mut value = Vec::with_capacity(chunks.len() * 32);
                for chunk in chunks {
                    value.extend_from_slice(chunk.as_bytes());
                }
                manifests.insert(manifest.as_bytes().as_slice(), value.as_slice())?;
                newly_recorded = true;
            }
        }
        if newly_recorded {
            let mut refcount = txn.open_table(REFCOUNT)?;
            for chunk in chunks {
                let key = chunk.as_bytes().as_slice();
                let current = refcount.get(key)?.map_or(0, |v| v.value());
                refcount.insert(key, current + 1)?;
            }
        }
        txn.commit()?;
        Ok(newly_recorded)
    }

    /// The ordered chunk ids of `manifest`, or `None` if it is unknown.
    ///
    /// # Errors
    /// Returns a [`MetaError`] on a table read failure or a corrupt manifest.
    pub fn get_manifest(&self, manifest: ContentId) -> Result<Option<Vec<ContentId>>, MetaError> {
        let txn = self.db.begin_read()?;
        let manifests = match txn.open_table(MANIFESTS) {
            Ok(table) => table,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(err) => return Err(err.into()),
        };
        let Some(value) = manifests.get(manifest.as_bytes().as_slice())? else {
            return Ok(None);
        };
        Ok(Some(decode_ids(value.value())?))
    }

    /// Anchor a live reference `name` (a tag) to `manifest`.
    ///
    /// # Errors
    /// Returns a [`MetaError`] if the write or commit fails.
    pub fn set_root(&self, name: &str, manifest: ContentId) -> Result<(), MetaError> {
        let txn = self.db.begin_write()?;
        {
            let mut roots = txn.open_table(ROOTS)?;
            roots.insert(name, manifest.as_bytes().as_slice())?;
        }
        txn.commit()?;
        Ok(())
    }

    /// The manifest anchored by root `name`, or `None` if no such root exists.
    ///
    /// # Errors
    /// Returns a [`MetaError`] on a table read failure, or [`MetaError::CorruptManifest`]
    /// if the stored value is not exactly 32 bytes.
    pub fn get_root(&self, name: &str) -> Result<Option<ContentId>, MetaError> {
        let txn = self.db.begin_read()?;
        let roots = match txn.open_table(ROOTS) {
            Ok(table) => table,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(err) => return Err(err.into()),
        };
        let Some(value) = roots.get(name)? else {
            return Ok(None);
        };
        let bytes: [u8; 32] = value
            .value()
            .try_into()
            .map_err(|_| MetaError::CorruptManifest(value.value().len()))?;
        Ok(Some(ContentId::from_bytes(bytes)))
    }

    /// Remove the live reference `name`.
    ///
    /// # Errors
    /// Returns a [`MetaError`] if the write or commit fails.
    pub fn remove_root(&self, name: &str) -> Result<(), MetaError> {
        let txn = self.db.begin_write()?;
        {
            let mut roots = txn.open_table(ROOTS)?;
            roots.remove(name)?;
        }
        txn.commit()?;
        Ok(())
    }

    /// The current reference count for `chunk` (0 if untracked).
    ///
    /// # Errors
    /// Returns a [`MetaError`] on a table read failure.
    pub fn refcount(&self, chunk: ContentId) -> Result<u64, MetaError> {
        let txn = self.db.begin_read()?;
        let refcount = match txn.open_table(REFCOUNT) {
            Ok(table) => table,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(0),
            Err(err) => return Err(err.into()),
        };
        Ok(refcount
            .get(chunk.as_bytes().as_slice())?
            .map_or(0, |v| v.value()))
    }

    /// Mark phase: every chunk reachable from a live root manifest.
    ///
    /// # Errors
    /// Returns a [`MetaError`] on a table read failure or a corrupt manifest.
    pub fn reachable_chunks(&self) -> Result<BTreeSet<[u8; 32]>, MetaError> {
        let txn = self.db.begin_read()?;
        let roots = match txn.open_table(ROOTS) {
            Ok(table) => table,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(BTreeSet::new()),
            Err(err) => return Err(err.into()),
        };
        let manifests = match txn.open_table(MANIFESTS) {
            Ok(table) => table,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(BTreeSet::new()),
            Err(err) => return Err(err.into()),
        };
        let mut reachable = BTreeSet::new();
        for entry in roots.iter()? {
            let (_name, manifest_id) = entry?;
            if let Some(value) = manifests.get(manifest_id.value())? {
                for id in decode_ids(value.value())? {
                    reachable.insert(*id.as_bytes());
                }
            }
        }
        Ok(reachable)
    }

    /// Sweep set: tracked chunks NOT reachable from any live root. The caller
    /// deletes these blobs from the content-addressed store.
    ///
    /// # Errors
    /// Returns a [`MetaError`] on a table read failure or a corrupt manifest.
    pub fn collectible_chunks(&self) -> Result<Vec<ContentId>, MetaError> {
        let reachable = self.reachable_chunks()?;
        let txn = self.db.begin_read()?;
        let refcount = match txn.open_table(REFCOUNT) {
            Ok(table) => table,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(err) => return Err(err.into()),
        };
        let mut collectible = Vec::new();
        for entry in refcount.iter()? {
            let (key, _count) = entry?;
            let bytes: [u8; 32] = key
                .value()
                .try_into()
                .map_err(|_| MetaError::CorruptManifest(key.value().len()))?;
            if !reachable.contains(&bytes) {
                collectible.push(ContentId::from_bytes(bytes));
            }
        }
        Ok(collectible)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flakecache_cas::ObjectKind;

    fn chunk(seed: &[u8]) -> ContentId {
        ContentId::compute(ObjectKind::Chunk, seed)
    }
    fn manifest(seed: &[u8]) -> ContentId {
        ContentId::compute(ObjectKind::Level1Manifest, seed)
    }

    #[test]
    fn refcount_dedup_reassembly_and_gc() -> Result<(), MetaError> {
        let dir = tempfile::tempdir().unwrap();
        let meta = MetaStore::open(dir.path().join("meta.redb"))?;

        let (c1, c2, c3) = (chunk(b"c1"), chunk(b"c2"), chunk(b"c3"));
        let m1 = manifest(b"m1"); // [c1, c2]
        let m2 = manifest(b"m2"); // [c2, c3]  (c2 shared -> deduped)

        assert!(meta.put_manifest(m1, &[c1, c2])?, "first insert is new");
        assert!(
            !meta.put_manifest(m1, &[c1, c2])?,
            "re-insert is idempotent"
        );
        assert!(meta.put_manifest(m2, &[c2, c3])?);

        assert_eq!(meta.refcount(c1)?, 1);
        assert_eq!(meta.refcount(c2)?, 2, "shared chunk is deduped");
        assert_eq!(meta.refcount(c3)?, 1);

        assert_eq!(meta.get_manifest(m1)?, Some(vec![c1, c2]));
        assert_eq!(meta.get_manifest(manifest(b"absent"))?, None);

        // No roots -> everything is collectible.
        assert_eq!(meta.collectible_chunks()?.len(), 3);

        // Root m1 keeps c1, c2 alive; c3 is collectible.
        meta.set_root("tag", m1)?;
        assert_eq!(meta.collectible_chunks()?, vec![c3]);

        // Root m2 keeps c2, c3 alive too -> nothing collectible.
        meta.set_root("tag2", m2)?;
        assert!(meta.collectible_chunks()?.is_empty());

        // Drop both roots -> all collectible again.
        meta.remove_root("tag")?;
        meta.remove_root("tag2")?;
        assert_eq!(meta.collectible_chunks()?.len(), 3);
        Ok(())
    }

    #[test]
    fn get_root_reads_back_the_anchored_manifest() -> Result<(), MetaError> {
        let dir = tempfile::tempdir().unwrap();
        let meta = MetaStore::open(dir.path().join("meta.redb"))?;
        let m = manifest(b"m");

        assert_eq!(meta.get_root("absent")?, None, "unset root is None");
        meta.set_root("tag", m)?;
        assert_eq!(
            meta.get_root("tag")?,
            Some(m),
            "reads back the exact manifest id"
        );
        meta.remove_root("tag")?;
        assert_eq!(meta.get_root("tag")?, None, "removed root is None again");
        Ok(())
    }

    #[test]
    fn survives_reopen() -> Result<(), MetaError> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("meta.redb");
        let c = chunk(b"persist");
        let m = manifest(b"persist");
        {
            let meta = MetaStore::open(&path)?;
            meta.put_manifest(m, &[c])?;
            meta.set_root("t", m)?;
        }
        let meta = MetaStore::open(&path)?;
        assert_eq!(meta.refcount(c)?, 1);
        assert_eq!(meta.get_manifest(m)?, Some(vec![c]));
        assert!(
            meta.collectible_chunks()?.is_empty(),
            "root keeps it alive across reopen"
        );
        Ok(())
    }
}
