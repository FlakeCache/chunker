//! Rustler NIF bindings for Elixir.
//! Enable via the `nif` feature to build the cdylib for Elixir.

#![allow(clippy::option_if_let_else)]

use rustler::types::binary::OwnedBinary;
use rustler::{Binary, Env, NifResult, ResourceArc};
use std::sync::Mutex;

use crate::{chunking, compression, hashing, signing};

mod atoms {
    rustler::atoms! {
        ok,
        signing_failed,
        invalid_signature,
        invalid_secret_key,
        invalid_public_key,
        invalid_signature_length,
        decode_error,
        verification_failed,
        invalid_base32,
        zstd_compression_failed,
        zstd_decompression_failed,
        xz_compression_failed,
        xz_decompression_failed,
        chunk_bounds_invalid,
        io_error,
        logging_init_failed,
        zero_length_chunk,
        invalid_chunking_options,
        busy,
    }
}

#[rustler::nif]
fn enable_logging(level: String) -> NifResult<rustler::Atom> {
    let filter = tracing_subscriber::EnvFilter::new(level);
    let subscriber = tracing_subscriber::fmt().with_env_filter(filter).finish();

    match tracing::subscriber::set_global_default(subscriber) {
        Ok(()) => Ok(atoms::ok()),
        Err(_) => Err(rustler::error::Error::Term(Box::new(
            atoms::logging_init_failed(),
        ))),
    }
}

fn binary_from_vec<'a>(env: Env<'a>, data: &[u8]) -> NifResult<Binary<'a>> {
    let mut owned = OwnedBinary::new(data.len())
        .ok_or_else(|| rustler::error::Error::RaiseTerm(Box::new("alloc_failed")))?;
    owned.as_mut_slice().copy_from_slice(data);
    Ok(owned.release(env))
}

// =============================================================================
// Signing Functions (Ed25519)
// =============================================================================

#[rustler::nif]
fn generate_keypair() -> NifResult<(String, String)> {
    let (secret_b64, public_b64) = signing::generate_keypair();
    Ok((secret_b64, public_b64))
}

#[rustler::nif]
fn sign_data<'a>(env: Env<'a>, data: Binary<'a>, secret_key_b64: &str) -> NifResult<String> {
    let _ = env;
    match signing::sign_data(data.as_slice(), secret_key_b64) {
        Ok(signature_b64) => Ok(signature_b64),
        Err(signing::SigningError::InvalidSecretKey) => Err(rustler::error::Error::Term(Box::new(
            atoms::invalid_secret_key(),
        ))),
        Err(signing::SigningError::DecodeError) => {
            Err(rustler::error::Error::Term(Box::new(atoms::decode_error())))
        }
        Err(_) => Err(rustler::error::Error::Term(Box::new(
            atoms::signing_failed(),
        ))),
    }
}

#[rustler::nif]
fn verify_signature<'a>(
    env: Env<'a>,
    data: Binary<'a>,
    signature_b64: &str,
    public_key_b64: &str,
) -> NifResult<rustler::Atom> {
    let _ = env;
    match signing::verify_signature(data.as_slice(), signature_b64, public_key_b64) {
        Ok(()) => Ok(atoms::ok()),
        Err(signing::SigningError::InvalidPublicKey) => Err(rustler::error::Error::Term(Box::new(
            atoms::invalid_public_key(),
        ))),
        Err(signing::SigningError::InvalidSecretKey) => Err(rustler::error::Error::Term(Box::new(
            atoms::invalid_secret_key(),
        ))),
        Err(signing::SigningError::InvalidSignature) => Err(rustler::error::Error::Term(Box::new(
            atoms::invalid_signature_length(),
        ))),
        Err(signing::SigningError::DecodeError) => {
            Err(rustler::error::Error::Term(Box::new(atoms::decode_error())))
        }
        Err(signing::SigningError::VerificationFailed) => Err(rustler::error::Error::Term(
            Box::new(atoms::invalid_signature()),
        )),
    }
}

// =============================================================================
// Hashing Functions (SHA256, Nix base32)
// =============================================================================

#[rustler::nif]
fn sha256_hash<'a>(env: Env<'a>, data: Binary<'a>) -> NifResult<String> {
    let _ = env;
    Ok(hashing::sha256_hash(data.as_slice()))
}

#[rustler::nif]
fn nix_base32_encode<'a>(env: Env<'a>, data: Binary<'a>) -> NifResult<String> {
    let _ = env;
    Ok(hashing::nix_base32_encode(data.as_slice()))
}

#[rustler::nif]
fn nix_base32_decode<'a>(env: Env<'a>, encoded: &str) -> NifResult<Binary<'a>> {
    match hashing::nix_base32_decode(encoded) {
        Ok(decoded) => binary_from_vec(env, &decoded),
        Err(_) => Err(rustler::error::Error::Term(Box::new(
            atoms::invalid_base32(),
        ))),
    }
}

// =============================================================================
// Compression Functions (zstd, xz)
// =============================================================================

#[rustler::nif(schedule = "DirtyCpu")]
fn compress_zstd<'a>(env: Env<'a>, data: Binary<'a>, level: Option<i32>) -> NifResult<Binary<'a>> {
    match compression::compress_zstd(data.as_slice(), level) {
        Ok(compressed) => binary_from_vec(env, &compressed),
        Err(_) => Err(rustler::error::Error::Term(Box::new(
            atoms::zstd_compression_failed(),
        ))),
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
fn decompress_zstd<'a>(env: Env<'a>, data: Binary<'a>) -> NifResult<Binary<'a>> {
    // Enforce a strict 256MB limit for NIF decompression to prevent OOMing the Erlang VM
    const NIF_DECOMPRESSION_LIMIT: u64 = 256 * 1024 * 1024;

    match compression::decompress_zstd_with_limit(data.as_slice(), NIF_DECOMPRESSION_LIMIT) {
        Ok(decompressed) => binary_from_vec(env, &decompressed),
        Err(compression::CompressionError::SizeExceeded | _) => Err(rustler::error::Error::Term(
            Box::new(atoms::zstd_decompression_failed()),
        )),
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
fn compress_xz<'a>(env: Env<'a>, data: Binary<'a>, level: Option<u32>) -> NifResult<Binary<'a>> {
    match compression::compress_xz(data.as_slice(), level) {
        Ok(compressed) => binary_from_vec(env, &compressed),
        Err(_) => Err(rustler::error::Error::Term(Box::new(
            atoms::xz_compression_failed(),
        ))),
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
fn decompress_xz<'a>(env: Env<'a>, data: Binary<'a>) -> NifResult<Binary<'a>> {
    match compression::decompress_xz(data.as_slice()) {
        Ok(decompressed) => binary_from_vec(env, &decompressed),
        Err(_) => Err(rustler::error::Error::Term(Box::new(
            atoms::xz_decompression_failed(),
        ))),
    }
}

// =============================================================================
// Chunking Functions (FastCDC)
// =============================================================================

#[rustler::nif(schedule = "DirtyCpu")]
fn chunk_data<'a>(
    env: Env<'a>,
    data: Binary<'a>,
    min_size: Option<u32>,
    avg_size: Option<u32>,
    max_size: Option<u32>,
) -> NifResult<Vec<(String, u64, u64)>> {
    let _ = env;

    match chunking::chunk_descriptors(
        data.as_slice(),
        min_size.map(|value| value as usize),
        avg_size.map(|value| value as usize),
        max_size.map(|value| value as usize),
    ) {
        Ok(chunks) => Ok(chunks
            .into_iter()
            .map(|chunk| (chunk.hash_hex(), chunk.offset, chunk.length as u64))
            .collect()),
        Err(error) => Err(map_chunking_error(&error)),
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
fn chunk_data_streaming<'a>(
    env: Env<'a>,
    data: Binary<'a>,
    min_size: Option<u32>,
    avg_size: Option<u32>,
    max_size: Option<u32>,
) -> NifResult<Vec<(String, u64, u64)>> {
    let _ = env;

    match chunking::chunk_descriptors(
        data.as_slice(),
        min_size.map(|value| value as usize),
        avg_size.map(|value| value as usize),
        max_size.map(|value| value as usize),
    ) {
        Ok(chunks) => Ok(chunks
            .into_iter()
            .map(|chunk| (chunk.hash_hex(), chunk.offset, chunk.length as u64))
            .collect()),
        Err(error) => Err(map_chunking_error(&error)),
    }
}

// =============================================================================
// Push-fed streaming chunking (bounded-memory ingest)
//
// The caller feeds upload-body slices as they arrive and receives the chunks
// whose boundaries are now final, each with its payload for inline storage. The
// chunker's retained internal state stays bounded by max_size regardless of
// total artifact size. Per-call transient memory is bounded by the SLICE size,
// not the artifact: the input binary, its buffered copy, and the freshly
// allocated output binaries can all be live at once while terms are encoded
// (~2-3x the slice). Callers should therefore feed bounded slices (a few MiB),
// which also yields natural backpressure via the synchronous
// read -> push -> store loop.
// =============================================================================

/// Opaque handle wrapping a [`chunking::PushChunker`] for one upload.
///
/// A single upload must be driven by ONE owner, sequentially
/// (`new` -> `push`* -> `finish`). The `Mutex` provides the interior mutability
/// the shared-reference NIF signature requires and guards against accidental
/// concurrent use: contending calls receive a `busy` error (via `try_lock`)
/// rather than silently interleaving the byte stream or parking a dirty
/// scheduler thread. Sequential use never contends.
#[derive(Debug)]
pub struct PushChunkerResource {
    inner: Mutex<chunking::PushChunker>,
}

#[rustler::resource_impl]
impl rustler::Resource for PushChunkerResource {}

/// Encode finalized chunks as `{hash_hex, offset, length, payload_binary}` tuples.
/// The returned binaries are freshly allocated in `env` (they do not borrow the
/// chunker's buffer), so the caller may store them and drop the resource freely.
fn chunks_to_terms(
    env: Env<'_>,
    chunks: Vec<chunking::ChunkMetadata>,
) -> NifResult<Vec<(String, u64, u64, Binary<'_>)>> {
    chunks
        .into_iter()
        .map(|chunk| {
            let payload = binary_from_vec(env, chunk.payload.as_ref())?;
            Ok((chunk.hash_hex(), chunk.offset, chunk.length as u64, payload))
        })
        .collect()
}

/// Acquire the per-upload chunker without parking a dirty scheduler thread.
/// Concurrent use of the same resource (a caller bug) yields `busy`; a poisoned
/// mutex (from a prior panic) yields `io_error`.
fn lock_chunker(
    resource: &PushChunkerResource,
) -> Result<std::sync::MutexGuard<'_, chunking::PushChunker>, rustler::error::Error> {
    match resource.inner.try_lock() {
        Ok(guard) => Ok(guard),
        Err(std::sync::TryLockError::WouldBlock) => {
            Err(rustler::error::Error::Term(Box::new(atoms::busy())))
        }
        Err(std::sync::TryLockError::Poisoned(_)) => {
            Err(rustler::error::Error::Term(Box::new(atoms::io_error())))
        }
    }
}

#[rustler::nif]
fn chunk_stream_new(
    min_size: Option<u32>,
    avg_size: Option<u32>,
    max_size: Option<u32>,
) -> NifResult<ResourceArc<PushChunkerResource>> {
    match chunking::PushChunker::new(
        min_size.map(|value| value as usize),
        avg_size.map(|value| value as usize),
        max_size.map(|value| value as usize),
        chunking::HashAlgorithm::Sha256,
    ) {
        Ok(chunker) => Ok(ResourceArc::new(PushChunkerResource {
            inner: Mutex::new(chunker),
        })),
        Err(error) => Err(map_chunking_error(&error)),
    }
}

// `resource` is passed by value because the NIF ABI decodes owned arguments; it
// cannot be a reference.
#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::needless_pass_by_value)]
fn chunk_stream_push<'a>(
    env: Env<'a>,
    resource: ResourceArc<PushChunkerResource>,
    data: Binary<'_>,
) -> NifResult<Vec<(String, u64, u64, Binary<'a>)>> {
    // Scope the lock so the guard is released before encoding NIF terms.
    let chunks = {
        let mut chunker = lock_chunker(&resource)?;
        chunker
            .push(data.as_slice())
            .map_err(|error| map_chunking_error(&error))?
    };
    chunks_to_terms(env, chunks)
}

#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::needless_pass_by_value)]
fn chunk_stream_finish(
    env: Env<'_>,
    resource: ResourceArc<PushChunkerResource>,
) -> NifResult<Vec<(String, u64, u64, Binary<'_>)>> {
    let chunks = {
        let mut chunker = lock_chunker(&resource)?;
        chunker
            .finish()
            .map_err(|error| map_chunking_error(&error))?
    };
    chunks_to_terms(env, chunks)
}

fn map_chunking_error(error: &chunking::ChunkingError) -> rustler::error::Error {
    match error {
        chunking::ChunkingError::Bounds { .. }
        | chunking::ChunkingError::BufferLimitExceeded { .. } => {
            rustler::error::Error::Term(Box::new(atoms::chunk_bounds_invalid()))
        }
        chunking::ChunkingError::Io(_) => rustler::error::Error::Term(Box::new(atoms::io_error())),
        chunking::ChunkingError::ZeroLengthChunk => {
            rustler::error::Error::Term(Box::new(atoms::zero_length_chunk()))
        }
        chunking::ChunkingError::InvalidOptions(_) => {
            rustler::error::Error::Term(Box::new(atoms::invalid_chunking_options()))
        }
    }
}
