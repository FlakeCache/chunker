// SPDX-License-Identifier: MIT
// Derived from rvf-crypto in RuVector; see THIRD_PARTY_NOTICES.md.

use core::fmt;

/// Errors returned while decoding or extending cryptographic structures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CryptoError {
    /// A serialized witness chain does not contain whole entries.
    InvalidWitnessLength,
    /// A witness entry does not bind to its predecessor.
    BrokenWitnessChain,
}

impl fmt::Display for CryptoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWitnessLength => formatter.write_str("invalid witness chain length"),
            Self::BrokenWitnessChain => formatter.write_str("broken witness chain"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CryptoError {}
