/// Rustler NIF bindings for Elixir
///
/// This module exposes the chunker library functions as NIF functions for use in Elixir.
/// It handles marshaling between Rust and Erlang types.

use rustler::{Atom, Binary, Env, NifResult, Term};

// Import the public library functions
use chunker::{chunking, compression, hashing, signing};

// Rustler NIF module initialization
// This macro generates the NIF bindings for Elixir
rustler::init!("Elixir.FlakecacheApp.Native.Chunker");

// ============================================================================
// Signing Functions (Ed25519)
// ============================================================================

#[rustler::nif]
fn generate_keypair() -> NifResult<(String, String)> {
    match signing::generate_keypair() {
        Ok((secret_b64, public_b64)) => Ok((secret_b64, public_b64)),
        Err(_) => Err(rustler::error::Error::Term(Box::new(Atom::new(
            "key_generation_failed",
        )))),
    }
}

#[rustler::nif]
fn sign_data<'a>(env: Env<'a>, data: Binary<'a>, secret_key_b64: &str) -> NifResult<String> {
    match signing::sign_data(data.as_slice(), secret_key_b64) {
        Ok(signature_b64) => Ok(signature_b64),
        Err(_) => Err(rustler::error::Error::Term(Box::new(Atom::new(
            "signing_failed",
        )))),
    }
}

#[rustler::nif]
fn verify_signature<'a>(
    data: Binary<'a>,
    signature_b64: &str,
    public_key_b64: &str,
) -> NifResult<Atom> {
    match signing::verify_signature(data.as_slice(), signature_b64, public_key_b64) {
        Ok(_) => Ok(Atom::new("ok")),
        Err(_) => Err(rustler::error::Error::Term(Box::new(Atom::new(
            "invalid_signature",
        )))),
    }
}

// ============================================================================
// Hashing Functions (SHA256, Nix base32)
// ============================================================================

#[rustler::nif]
fn sha256_hash<'a>(env: Env<'a>, data: Binary<'a>) -> NifResult<String> {
    Ok(hashing::sha256_hash(data.as_slice()))
}

#[rustler::nif]
fn nix_base32_encode<'a>(env: Env<'a>, data: Binary<'a>) -> NifResult<String> {
    Ok(hashing::nix_base32_encode(data.as_slice()))
}

#[rustler::nif]
fn nix_base32_decode(encoded: &str) -> NifResult<Binary> {
    match hashing::nix_base32_decode(encoded) {
        Ok(decoded) => Ok(Binary::from(decoded)),
        Err(_) => Err(rustler::error::Error::Term(Box::new(Atom::new(
            "invalid_base32",
        )))),
    }
}

// ============================================================================
// Compression Functions (zstd, xz, bzip2)
// ============================================================================

#[rustler::nif]
fn compress_zstd<'a>(env: Env<'a>, data: Binary<'a>, level: i32) -> NifResult<Binary> {
    match compression::compress_zstd(data.as_slice(), level as u32) {
        Ok(compressed) => Ok(Binary::from(compressed)),
        Err(_) => Err(rustler::error::Error::Term(Box::new(Atom::new(
            "zstd_compression_failed",
        )))),
    }
}

#[rustler::nif]
fn decompress_zstd<'a>(env: Env<'a>, data: Binary<'a>) -> NifResult<Binary> {
    match compression::decompress_zstd(data.as_slice()) {
        Ok(decompressed) => Ok(Binary::from(decompressed)),
        Err(_) => Err(rustler::error::Error::Term(Box::new(Atom::new(
            "zstd_decompression_failed",
        )))),
    }
}

#[rustler::nif]
fn compress_xz<'a>(env: Env<'a>, data: Binary<'a>, level: i32) -> NifResult<Binary> {
    match compression::compress_xz(data.as_slice(), level as u32) {
        Ok(compressed) => Ok(Binary::from(compressed)),
        Err(_) => Err(rustler::error::Error::Term(Box::new(Atom::new(
            "xz_compression_failed",
        )))),
    }
}

#[rustler::nif]
fn decompress_xz<'a>(env: Env<'a>, data: Binary<'a>) -> NifResult<Binary> {
    match compression::decompress_xz(data.as_slice()) {
        Ok(decompressed) => Ok(Binary::from(decompressed)),
        Err(_) => Err(rustler::error::Error::Term(Box::new(Atom::new(
            "xz_decompression_failed",
        )))),
    }
}

// ============================================================================
// Chunking Functions (FastCDC)
// ============================================================================

#[rustler::nif]
fn chunk_data<'a>(
    env: Env<'a>,
    data: Binary<'a>,
    min_size: Option<u32>,
    avg_size: Option<u32>,
    max_size: Option<u32>,
) -> NifResult<Vec<(String, u32, u32)>> {
    let min = min_size.unwrap_or(16_384);
    let avg = avg_size.unwrap_or(65_536);
    let max = max_size.unwrap_or(262_144);

    match chunking::chunk_data(data.as_slice(), min, avg, max) {
        Ok(chunks) => Ok(chunks
            .into_iter()
            .map(|(hash, offset, length)| (hash, offset, length))
            .collect()),
        Err(_) => Err(rustler::error::Error::Term(Box::new(Atom::new(
            "chunking_failed",
        )))),
    }
}
