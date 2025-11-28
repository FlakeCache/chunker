//! # Chunker
//!
//! High-performance content-defined chunking (`FastCDC`) for Nix NARs.
//!
//! This crate provides:
//! - **`FastCDC`** content-defined chunking algorithm
//! - **Compression** codecs (zstd, xz, bzip2)
//! - **Cryptographic signing** (Ed25519)
//! - **Hash computation** (SHA256, Nix base32)
//!
//! ## Observability & Telemetry
//!
//! This crate uses the [`tracing`](https://docs.rs/tracing) ecosystem, making it **Telemetry Ready**.
//!
//! ### Rust Applications
//! You can export traces to **Jaeger**, **Datadog**, or **Honeycomb** by installing an OpenTelemetry subscriber in your application binary:
//!
//! ```rust,ignore
//! // In your main.rs
//! use tracing_subscriber::prelude::*;
//!
//! let tracer = opentelemetry_jaeger::new_pipeline()
//!     .with_service_name("chunker")
//!     .install_simple()
//!     .unwrap();
//!
//! let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);
//! tracing_subscriber::registry().with(telemetry).init();
//! ```
//!
//! ### Elixir Applications
//! The NIF exposes `enable_logging/1` for runtime debugging. For full distributed tracing,
//! you can initialize a Rust-side OpenTelemetry subscriber that correlates with your Elixir traces.
//!
//! ## Example
//!
//! ```rust
//! use chunker::chunking;
//! use std::io::Cursor;
//!
//! let data = vec![0u8; 1024 * 1024]; // 1MB data
//! let chunks = chunking::chunk_data(&data, None, None, None).unwrap();
//!
//! println!("Generated {} chunks", chunks.len());
//! ```
//!
//! When compiled with the `nif` feature, provides Rustler NIF bindings for Elixir.

#[cfg(all(feature = "nif", feature = "telemetry"))]
compile_error!(
    "The 'nif' and 'telemetry' features are mutually exclusive. \
    Enabling 'telemetry' (Tokio runtime) inside a NIF is unsafe and can crash the Erlang VM. \
    Use standard logging for NIFs instead."
);

pub mod chunking;
pub mod compression;
pub mod hashing;
pub mod signing;

pub use chunking::{ChunkMetadata, ChunkStream, ChunkingOptions, HashAlgorithm};

#[cfg(feature = "nif")]
pub mod nif;

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn test_sha256_basic() {
        let data = b"hello world";
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        let hash_hex = hex::encode(result);

        // Should be 64 characters
        assert_eq!(hash_hex.len(), 64);
        // Known hash for "hello world"
        assert_eq!(
            hash_hex,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_nix_base32_alphabet() {
        // Test that our alphabet constant is correct
        let alphabet = hashing::NIX_BASE32_ALPHABET;
        assert_eq!(alphabet.len(), 32);
        assert_eq!(alphabet[0], b'0');
        assert_eq!(alphabet[31], b'z');
    }

    #[test]
    fn test_zstd_compression_basic() -> Result<(), Box<dyn std::error::Error>> {
        let data = b"hello world test data for compression that should be long enough to actually compress";

        // Test compression/decompression
        let compressed = zstd::encode_all(data.as_slice(), 3)?;
        let decompressed = zstd::decode_all(compressed.as_slice())?;

        assert_eq!(decompressed, data);
        // Note: For small data, compressed might not be smaller due to overhead
        Ok(())
    }

    #[test]
    fn test_ed25519_key_generation() {
        use ed25519_dalek::{Signer, SigningKey, Verifier};

        let mut key_bytes = [0u8; 32];
        key_bytes.fill(0);
        let signing_key = SigningKey::from_bytes(&key_bytes);
        let verifying_key = signing_key.verifying_key();

        // Keys should be correct length
        assert_eq!(signing_key.as_bytes().len(), 32);
        assert_eq!(verifying_key.as_bytes().len(), 32);

        // Test signing
        let message = b"test message";
        let signature = signing_key.sign(message);

        // Test verification
        let result = verifying_key.verify(message, &signature);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fastcdc_chunking_basic() {
        use fastcdc::v2020::FastCDC;

        let data = b"hello world this is some test data for chunking algorithm";
        let chunker = FastCDC::new(data, 16_384, 65_536, 262_144);

        let chunks: Vec<_> = chunker.collect();
        assert!(!chunks.is_empty());

        // Verify chunks cover the entire data
        let mut total_length = 0;
        for chunk in &chunks {
            assert!(chunk.length > 0);
            assert!(chunk.offset + chunk.length <= data.len());
            total_length += chunk.length;
        }
        assert_eq!(total_length, data.len());
    }

    #[test]
    fn test_chunk_hash_calculation() {
        let data = b"test chunk";

        let mut hasher = Sha256::new();
        hasher.update(data);
        let hash = hasher.finalize();
        let hash_hex = hex::encode(hash);

        assert_eq!(hash_hex.len(), 64);
        assert!(hash_hex.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
