// SPDX-License-Identifier: MIT
// Derived from rvf-crypto in RuVector; see THIRD_PARTY_NOTICES.md.

//! SHAKE-256-linked witness entries for tamper-evident append logs.

use alloc::vec::Vec;

use crate::{CryptoError, shake256_256};

/// Serialized size of one witness entry.
pub const WITNESS_ENTRY_SIZE: usize = 73;

/// A single event in a witness chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WitnessEntry {
    /// SHAKE-256-256 of the preceding serialized entry, or zero for genesis.
    pub prev_hash: [u8; 32],
    /// Hash of the operation, object, or transition being witnessed.
    pub action_hash: [u8; 32],
    /// Nanosecond UNIX timestamp supplied by the caller.
    pub timestamp_ns: u64,
    /// Application-defined event type.
    pub witness_type: u8,
}

impl WitnessEntry {
    /// Construct an unlinked entry. Append operations set `prev_hash`.
    #[must_use]
    pub const fn new(action_hash: [u8; 32], timestamp_ns: u64, witness_type: u8) -> Self {
        Self {
            prev_hash: [0_u8; 32],
            action_hash,
            timestamp_ns,
            witness_type,
        }
    }
}

fn encode_entry(entry: &WitnessEntry) -> [u8; WITNESS_ENTRY_SIZE] {
    let mut bytes = [0_u8; WITNESS_ENTRY_SIZE];
    bytes[..32].copy_from_slice(&entry.prev_hash);
    bytes[32..64].copy_from_slice(&entry.action_hash);
    bytes[64..72].copy_from_slice(&entry.timestamp_ns.to_le_bytes());
    bytes[72] = entry.witness_type;
    bytes
}

fn decode_entry(bytes: &[u8]) -> Result<WitnessEntry, CryptoError> {
    let bytes: &[u8; WITNESS_ENTRY_SIZE] = bytes
        .try_into()
        .map_err(|_| CryptoError::InvalidWitnessLength)?;

    let mut prev_hash = [0_u8; 32];
    prev_hash.copy_from_slice(&bytes[..32]);
    let mut action_hash = [0_u8; 32];
    action_hash.copy_from_slice(&bytes[32..64]);

    Ok(WitnessEntry {
        prev_hash,
        action_hash,
        timestamp_ns: u64::from_le_bytes(
            bytes[64..72]
                .try_into()
                .map_err(|_| CryptoError::InvalidWitnessLength)?,
        ),
        witness_type: bytes[72],
    })
}

/// Append and link one witness entry to a serialized chain.
///
/// The existing bytes must end on an entry boundary. The returned hash is the
/// hash of the appended serialized entry and can be stored as a chain head.
///
/// # Errors
///
/// Returns [`CryptoError::InvalidWitnessLength`] if `chain` ends with a
/// partial entry.
pub fn append_witness_entry(
    chain: &mut Vec<u8>,
    entry: &WitnessEntry,
) -> Result<[u8; 32], CryptoError> {
    if chain.len() % WITNESS_ENTRY_SIZE != 0 {
        return Err(CryptoError::InvalidWitnessLength);
    }

    let prev_hash = chain
        .last_chunk::<WITNESS_ENTRY_SIZE>()
        .map_or([0_u8; 32], |previous| shake256_256(previous));
    let mut linked = entry.clone();
    linked.prev_hash = prev_hash;
    let encoded = encode_entry(&linked);
    let head = shake256_256(&encoded);
    chain.extend_from_slice(&encoded);
    Ok(head)
}

/// Link a sequence of entries into a newly allocated witness chain.
#[must_use]
pub fn create_witness_chain(entries: &[WitnessEntry]) -> Vec<u8> {
    let mut chain = Vec::with_capacity(entries.len() * WITNESS_ENTRY_SIZE);
    let mut prev_hash = [0_u8; 32];
    for entry in entries {
        let mut linked = entry.clone();
        linked.prev_hash = prev_hash;
        let encoded = encode_entry(&linked);
        prev_hash = shake256_256(&encoded);
        chain.extend_from_slice(&encoded);
    }
    chain
}

/// Verify every predecessor link and decode a serialized witness chain.
///
/// # Errors
///
/// Returns [`CryptoError::InvalidWitnessLength`] for a partial entry or
/// [`CryptoError::BrokenWitnessChain`] when a predecessor hash does not match.
pub fn verify_witness_chain(data: &[u8]) -> Result<Vec<WitnessEntry>, CryptoError> {
    if data.len() % WITNESS_ENTRY_SIZE != 0 {
        return Err(CryptoError::InvalidWitnessLength);
    }

    let mut expected_prev = [0_u8; 32];
    let mut entries = Vec::with_capacity(data.len() / WITNESS_ENTRY_SIZE);
    for entry_bytes in data.chunks_exact(WITNESS_ENTRY_SIZE) {
        let entry = decode_entry(entry_bytes)?;
        if entry.prev_hash != expected_prev {
            return Err(CryptoError::BrokenWitnessChain);
        }
        expected_prev = shake256_256(entry_bytes);
        entries.push(entry);
    }
    Ok(entries)
}
