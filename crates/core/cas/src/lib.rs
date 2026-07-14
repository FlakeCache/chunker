// SPDX-License-Identifier: MIT
//! Content-addressed store (CAS) for `FlakeCache`.
//!
//! Immutable `put(kind, bytes)` -> [`ContentId`] and `get(kind, id)` ->
//! `Option<Bytes>` over a pluggable [`BlobBackend`]. Content identifiers are
//! domain-separated SHAKE-256 digests per
//! `docs/design/content-addressed-store-format.md` (§3.1). Every read recomputes
//! and verifies the identifier before returning bytes, so a corrupt or
//! substituted blob is rejected rather than served.
//!
//! [`FilesystemBackend`] is the warm-tier foundation: objects live on a local
//! disk, sharded `objects/<2hex>/<rest>`, written crash-safely via temp+rename.
//! Cold-tier (S3 / Storage Box) write-through and hot/cold tiering layer on top
//! of the same [`BlobBackend`] trait. Whole-blob-on-S3 is never a storage mode:
//! callers store `FastCDC` chunks addressed by content hash.

use std::fs;
use std::io;
use std::path::PathBuf;

use bytes::Bytes;

/// Domain prefix bound into every content id (spec §3.1).
const CAS_DOMAIN: &[u8] = b"flakecache-cas";
/// Content-id format version bound into the hash input.
const FORMAT_VERSION: u16 = 1;

/// Object kinds that participate in content-id domain separation (format v1).
///
/// The kind is part of the hash input, so the same bytes stored as two
/// different kinds get two different identifiers and can never be confused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ObjectKind {
    /// A `FastCDC` chunk of a NAR/layer byte stream.
    Chunk = 0x01,
    /// Fixed-size bootstrap manifest (spec §4.1).
    Level0Manifest = 0x02,
    /// Full TLV manifest (spec §4.2).
    Level1Manifest = 0x03,
    /// Reference-count segment.
    Refcount = 0x04,
    /// Witness (hash-chain) segment.
    Witness = 0x05,
    /// Object recipe / index.
    Recipe = 0x06,
    /// Immutable commit record.
    Commit = 0x07,
}

/// A 32-byte content identifier.
///
/// `id = SHAKE256-256(domain || 0x00 || kind || format_version || len || payload)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentId([u8; 32]);

impl ContentId {
    /// Compute the identifier for `payload` under `kind`.
    #[must_use]
    pub fn compute(kind: ObjectKind, payload: &[u8]) -> Self {
        let mut input = Vec::with_capacity(CAS_DOMAIN.len() + 12 + payload.len());
        input.extend_from_slice(CAS_DOMAIN);
        input.push(0x00);
        input.push(kind as u8);
        input.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        input.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        input.extend_from_slice(payload);
        Self(flakecache_crypto::shake256_256(&input))
    }

    /// The raw 32-byte digest.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Lowercase hex encoding (64 characters).
    #[must_use]
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }
}

/// Errors from the content-addressed store.
#[derive(Debug, thiserror::Error)]
pub enum CasError {
    /// A backend I/O operation failed.
    #[error("backend io error: {0}")]
    Io(#[from] io::Error),
    /// Stored bytes did not hash to the requested identifier.
    #[error("integrity check failed: expected {expected}, computed {actual}")]
    Integrity {
        /// The identifier that was requested.
        expected: String,
        /// The identifier the returned bytes actually hash to.
        actual: String,
    },
}

/// A minimal blob backend: store and fetch immutable bytes keyed by [`ContentId`].
///
/// `put` MUST be idempotent (re-storing identical content is a no-op) and blobs
/// are immutable once written. Identifier/content verification is performed by
/// [`Cas`] above this trait, because verification needs the object [`ObjectKind`]
/// that framed the hash and a bare backend does not carry it.
pub trait BlobBackend {
    /// Store `bytes` under `id`. Idempotent.
    ///
    /// # Errors
    /// Returns [`CasError::Io`] if the backend write fails.
    fn put(&self, id: ContentId, bytes: &[u8]) -> Result<(), CasError>;

    /// Fetch the bytes stored under `id`, or `None` if absent.
    ///
    /// # Errors
    /// Returns [`CasError::Io`] if the backend read fails.
    fn get(&self, id: ContentId) -> Result<Option<Bytes>, CasError>;
}

/// A local-filesystem warm-tier backend.
///
/// Objects are sharded as `objects/<first-two-hex>/<remaining-hex>` under a root
/// directory and written crash-safely (temp file + atomic rename). Because blobs
/// are immutable and content-addressed, an existing object is never rewritten.
#[derive(Debug, Clone)]
pub struct FilesystemBackend {
    root: PathBuf,
}

impl FilesystemBackend {
    /// Create a backend rooted at `root` (created lazily on first `put`).
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn object_path(&self, id: ContentId) -> PathBuf {
        let hex = id.to_hex();
        self.root.join("objects").join(&hex[..2]).join(&hex[2..])
    }
}

impl BlobBackend for FilesystemBackend {
    fn put(&self, id: ContentId, bytes: &[u8]) -> Result<(), CasError> {
        let path = self.object_path(id);
        // Immutable + idempotent: an object present under this id already holds
        // exactly these bytes (the caller verified the id), so do nothing.
        if path.exists() {
            return Ok(());
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        // Crash-safety: write to a temp sibling then atomically rename into place.
        // The temp is per-id, so a concurrent writer of the same id writes the
        // same bytes and the rename remains correct.
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, bytes)?;
        match fs::rename(&tmp, &path) {
            Ok(()) => Ok(()),
            Err(err) => {
                let _ = fs::remove_file(&tmp);
                Err(err.into())
            }
        }
    }

    fn get(&self, id: ContentId) -> Result<Option<Bytes>, CasError> {
        match fs::read(self.object_path(id)) {
            Ok(bytes) => Ok(Some(Bytes::from(bytes))),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err.into()),
        }
    }
}

/// The content-addressed store: computes and verifies identifiers around a
/// [`BlobBackend`].
#[derive(Debug, Clone)]
pub struct Cas<B: BlobBackend> {
    backend: B,
}

impl<B: BlobBackend> Cas<B> {
    /// Wrap a backend.
    pub const fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Store `payload` under `kind` and return its content id. Idempotent.
    ///
    /// # Errors
    /// Returns [`CasError::Io`] if the backend write fails.
    pub fn put(&self, kind: ObjectKind, payload: &[u8]) -> Result<ContentId, CasError> {
        let id = ContentId::compute(kind, payload);
        self.backend.put(id, payload)?;
        Ok(id)
    }

    /// Fetch the payload for `id` (framed as `kind`), verifying it hashes to `id`.
    ///
    /// # Errors
    /// Returns [`CasError::Integrity`] if the stored bytes do not hash to `id`,
    /// or [`CasError::Io`] on a backend read failure.
    pub fn get(&self, kind: ObjectKind, id: ContentId) -> Result<Option<Bytes>, CasError> {
        let Some(bytes) = self.backend.get(id)? else {
            return Ok(None);
        };
        let actual = ContentId::compute(kind, &bytes);
        if actual == id {
            Ok(Some(bytes))
        } else {
            Err(CasError::Integrity {
                expected: id.to_hex(),
                actual: actual.to_hex(),
            })
        }
    }

    /// Borrow the underlying backend.
    pub const fn backend(&self) -> &B {
        &self.backend
    }
}

/// A convenience alias for a CAS over the local warm-tier filesystem backend.
pub type FsCas = Cas<FilesystemBackend>;

impl FsCas {
    /// Open a filesystem-backed CAS rooted at `root`.
    pub fn open(root: impl Into<PathBuf>) -> Self {
        Cas::new(FilesystemBackend::new(root))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_is_deterministic_and_kind_separated() {
        let a = ContentId::compute(ObjectKind::Chunk, b"hello");
        let b = ContentId::compute(ObjectKind::Chunk, b"hello");
        let c = ContentId::compute(ObjectKind::Refcount, b"hello");
        assert_eq!(a, b, "same kind+bytes must be the same id");
        assert_ne!(a, c, "different kind must change the id");
        assert_eq!(a.to_hex().len(), 64);
    }

    #[test]
    fn put_get_round_trip_and_dedup() -> Result<(), CasError> {
        let dir = tempfile::tempdir().unwrap();
        let cas = FsCas::open(dir.path());

        let id1 = cas.put(ObjectKind::Chunk, b"some chunk bytes")?;
        // Idempotent re-put of identical content yields the same id, no error.
        let id2 = cas.put(ObjectKind::Chunk, b"some chunk bytes")?;
        assert_eq!(id1, id2);

        let got = cas.get(ObjectKind::Chunk, id1)?;
        assert_eq!(got.as_deref(), Some(&b"some chunk bytes"[..]));

        // Absent id -> None.
        let missing = ContentId::compute(ObjectKind::Chunk, b"never stored");
        assert!(cas.get(ObjectKind::Chunk, missing)?.is_none());
        Ok(())
    }

    #[test]
    fn get_rejects_tampered_object() {
        let dir = tempfile::tempdir().unwrap();
        let cas = FsCas::open(dir.path());
        let id = cas.put(ObjectKind::Chunk, b"trusted payload").unwrap();

        // Corrupt the stored object on disk under the same id.
        let path = cas.backend().object_path(id);
        std::fs::write(&path, b"tampered payload").unwrap();

        match cas.get(ObjectKind::Chunk, id) {
            Err(CasError::Integrity { .. }) => {}
            other => panic!("expected Integrity error, got {other:?}"),
        }
    }

    #[test]
    fn get_with_wrong_kind_fails_verification() {
        let dir = tempfile::tempdir().unwrap();
        let cas = FsCas::open(dir.path());
        let id = cas.put(ObjectKind::Chunk, b"payload").unwrap();
        // The object exists at `id`, but re-framing the same bytes under a
        // different kind recomputes a different id, so verification rejects it.
        match cas.get(ObjectKind::Refcount, id) {
            Err(CasError::Integrity { .. }) => {}
            other => panic!("expected Integrity error for wrong kind, got {other:?}"),
        }
    }
}
