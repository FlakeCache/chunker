// SPDX-License-Identifier: MIT
// Derived from rvf-crypto in RuVector; see THIRD_PARTY_NOTICES.md.

//! Generic Ed25519 signing and verification.
//!
//! Callers are responsible for defining a canonical message and including an
//! application-specific domain separator before signing it.

use ed25519_dalek::{Signature, Signer};
pub use ed25519_dalek::{SigningKey, VerifyingKey};

/// Reconstruct an Ed25519 signing key from its 32-byte seed.
///
/// Nix binary-cache secret keys store the 32-byte seed followed by the 32-byte
/// public key; this takes the seed half.
#[must_use]
pub fn signing_key_from_seed(seed: &[u8; 32]) -> SigningKey {
    SigningKey::from_bytes(seed)
}

/// Sign an already-canonicalized message with Ed25519.
#[must_use]
pub fn sign_message(message: &[u8], signing_key: &SigningKey) -> [u8; 64] {
    signing_key.sign(message).to_bytes()
}

/// Verify an Ed25519 signature over an already-canonicalized message.
#[must_use]
pub fn verify_message(message: &[u8], signature: &[u8; 64], verifying_key: &VerifyingKey) -> bool {
    verifying_key
        .verify_strict(message, &Signature::from_bytes(signature))
        .is_ok()
}
