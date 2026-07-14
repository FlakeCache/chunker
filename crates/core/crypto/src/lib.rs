// SPDX-License-Identifier: MIT
// Derived from rvf-crypto in RuVector; see THIRD_PARTY_NOTICES.md.

//! Cryptographic primitives for `FlakeCache`.
//!
//! The crate is `no_std` outside the optional `std` feature and uses `alloc`
//! for variable-length hashes and serialized witness chains.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod b64;
mod error;
mod hash;
#[cfg(feature = "ed25519")]
mod sign;
mod witness;

pub use error::CryptoError;
pub use hash::{shake256_128, shake256_256, shake256_hash};
#[cfg(feature = "ed25519")]
pub use sign::{SigningKey, VerifyingKey, sign_message, signing_key_from_seed, verify_message};
pub use witness::{
    WITNESS_ENTRY_SIZE, WitnessEntry, append_witness_entry, create_witness_chain,
    verify_witness_chain,
};
