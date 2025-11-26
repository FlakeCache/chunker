//! Rustler NIF bindings for Elixir.
//! Enable via the `nif` feature to build the cdylib for Elixir integration.

use rustler::types::binary::OwnedBinary;
use rustler::{Binary, Env, NifResult};

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
    }
}

// Rustler NIF module initialization
rustler::init!(
    "Elixir.FlakecacheApp.Native.Chunker",
    [
        generate_keypair,
        sign_data,
        verify_signature,
        sha256_hash,
        nix_base32_encode,
        nix_base32_decode,
        compress_zstd,
        decompress_zstd,
        compress_xz,
        decompress_xz,
        chunk_data
    ]
);
fn binary_from_vec<'a>(env: Env<'a>, data: Vec<u8>) -> NifResult<Binary<'a>> {
    let mut owned = OwnedBinary::new(data.len())
        .ok_or_else(|| rustler::error::Error::RaiseTerm(Box::new("alloc_failed")))?;
    owned.as_mut_slice().copy_from_slice(&data);
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
        Ok(_) => Ok(atoms::ok()),
        Err(signing::SigningError::InvalidPublicKey) => Err(rustler::error::Error::Term(Box::new(
            atoms::invalid_public_key(),
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
fn nix_base32_encode<'a>(_env: Env<'a>, data: Binary<'a>) -> NifResult<String> {
    Ok(hashing::nix_base32_encode(data.as_slice()))
}

#[rustler::nif]
fn nix_base32_decode<'a>(env: Env<'a>, encoded: &str) -> NifResult<Binary<'a>> {
    match hashing::nix_base32_decode(encoded) {
        Ok(decoded) => binary_from_vec(env, decoded),
        Err(_) => Err(rustler::error::Error::Term(Box::new(
            atoms::invalid_base32(),
        ))),
    }
}

// =============================================================================
// Compression Functions (zstd, xz)
// =============================================================================

#[rustler::nif]
fn compress_zstd<'a>(env: Env<'a>, data: Binary<'a>, level: Option<i32>) -> NifResult<Binary<'a>> {
    match compression::compress_zstd(data.as_slice(), level) {
        Ok(compressed) => binary_from_vec(env, compressed),
        Err(_) => Err(rustler::error::Error::Term(Box::new(
            atoms::zstd_compression_failed(),
        ))),
    }
}

#[rustler::nif]
fn decompress_zstd<'a>(env: Env<'a>, data: Binary<'a>) -> NifResult<Binary<'a>> {
    match compression::decompress_zstd(data.as_slice()) {
        Ok(decompressed) => binary_from_vec(env, decompressed),
        Err(_) => Err(rustler::error::Error::Term(Box::new(
            atoms::zstd_decompression_failed(),
        ))),
    }
}

#[rustler::nif]
fn compress_xz<'a>(env: Env<'a>, data: Binary<'a>, level: Option<u32>) -> NifResult<Binary<'a>> {
    match compression::compress_xz(data.as_slice(), level) {
        Ok(compressed) => binary_from_vec(env, compressed),
        Err(_) => Err(rustler::error::Error::Term(Box::new(
            atoms::xz_compression_failed(),
        ))),
    }
}

#[rustler::nif]
fn decompress_xz<'a>(env: Env<'a>, data: Binary<'a>) -> NifResult<Binary<'a>> {
    match compression::decompress_xz(data.as_slice()) {
        Ok(decompressed) => binary_from_vec(env, decompressed),
        Err(_) => Err(rustler::error::Error::Term(Box::new(
            atoms::xz_decompression_failed(),
        ))),
    }
}

// =============================================================================
// Chunking Functions (FastCDC)
// =============================================================================

#[rustler::nif]
fn chunk_data<'a>(
    _env: Env<'a>,
    data: Binary<'a>,
    min_size: Option<u32>,
    avg_size: Option<u32>,
    max_size: Option<u32>,
) -> NifResult<Vec<(String, u32, u32)>> {
    let min = min_size.unwrap_or(16_384) as usize;
    let avg = avg_size.unwrap_or(65_536) as usize;
    let max = max_size.unwrap_or(262_144) as usize;

    let cursor = std::io::Cursor::new(data.as_slice());

    match chunking::chunk_stream(cursor, Some(min), Some(avg), Some(max)) {
        Ok(chunks) => Ok(chunks
            .into_iter()
            .map(|(hash, offset, length)| {
                #[allow(clippy::cast_possible_truncation)]
                let offset_u32 = offset as u32;
                #[allow(clippy::cast_possible_truncation)]
                let length_u32 = length as u32;
                (hash, offset_u32, length_u32)
            })
            .collect()),
        Err(chunking::ChunkingError::Bounds { .. }) => Err(rustler::error::Error::Term(Box::new(
            atoms::chunk_bounds_invalid(),
        ))),
    }
}
