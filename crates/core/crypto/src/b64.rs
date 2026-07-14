// SPDX-License-Identifier: MIT
//! Standard base64 (RFC 4648, padded).
//!
//! Nix advertises Ed25519 public keys and narinfo signatures in padded standard
//! base64, so `FlakeCache` needs to encode signatures and decode secret keys in
//! exactly that form. Hand-rolled to keep the crate free of an external base64
//! dependency (see the workspace no-dependency policy).

use alloc::string::String;
use alloc::vec::Vec;

const ENC: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Errors from [`decode`].
#[derive(Debug, PartialEq, Eq)]
pub enum Base64Error {
    /// The input length is not a multiple of 4.
    Length,
    /// The input contains a byte outside the base64 alphabet.
    Char,
    /// Padding (`=`) appears anywhere but the final one or two positions.
    Padding,
}

impl core::fmt::Display for Base64Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Length => f.write_str("base64: length not a multiple of 4"),
            Self::Char => f.write_str("base64: byte outside the alphabet"),
            Self::Padding => f.write_str("base64: misplaced padding"),
        }
    }
}

impl core::error::Error for Base64Error {}

/// Encode `input` as padded standard base64.
#[must_use]
pub fn encode(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);
        out.push(ENC[(b0 >> 2) as usize] as char);
        out.push(ENC[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            out.push(ENC[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(ENC[(b2 & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// Decode padded standard base64.
///
/// # Errors
/// Returns [`Base64Error`] if the length is not a multiple of 4, a byte is
/// outside the alphabet, or padding is misplaced.
pub fn decode(input: &str) -> Result<Vec<u8>, Base64Error> {
    let bytes = input.as_bytes();
    if bytes.len() % 4 != 0 {
        return Err(Base64Error::Length);
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for quad in bytes.chunks(4) {
        let mut vals = [0u8; 4];
        let mut pad = 0u8;
        for (k, &b) in quad.iter().enumerate() {
            if b == b'=' {
                // Padding is only ever the last one or two characters.
                if k < 2 {
                    return Err(Base64Error::Padding);
                }
                pad += 1;
            } else {
                // A data byte after padding has begun is malformed.
                if pad > 0 {
                    return Err(Base64Error::Padding);
                }
                vals[k] = dec_val(b)?;
            }
        }
        let n = (u32::from(vals[0]) << 18)
            | (u32::from(vals[1]) << 12)
            | (u32::from(vals[2]) << 6)
            | u32::from(vals[3]);
        // `n` packs exactly 24 bits; each byte extraction is a deliberate
        // low-8-bit take, never a lossy narrowing of a wider value.
        #[allow(clippy::cast_possible_truncation)]
        {
            out.push((n >> 16) as u8);
            if pad < 2 {
                out.push((n >> 8) as u8);
            }
            if pad < 1 {
                out.push(n as u8);
            }
        }
    }
    Ok(out)
}

fn dec_val(b: u8) -> Result<u8, Base64Error> {
    match b {
        b'A'..=b'Z' => Ok(b - b'A'),
        b'a'..=b'z' => Ok(b - b'a' + 26),
        b'0'..=b'9' => Ok(b - b'0' + 52),
        b'+' => Ok(62),
        b'/' => Ok(63),
        _ => Err(Base64Error::Char),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_all_pad_lengths() {
        for input in [&b""[..], b"f", b"fo", b"foo", b"foob", b"fooba", b"foobar"] {
            assert_eq!(decode(&encode(input)).unwrap(), input);
        }
    }

    #[test]
    fn matches_known_vectors() {
        // RFC 4648 test vectors.
        assert_eq!(encode(b"foobar"), "Zm9vYmFy");
        assert_eq!(encode(b"fo"), "Zm8=");
        assert_eq!(encode(b"f"), "Zg==");
        assert_eq!(decode("Zm9vYmFy").unwrap(), b"foobar");
    }

    #[test]
    fn rejects_malformed() {
        assert_eq!(decode("abc"), Err(Base64Error::Length)); // not a multiple of 4
        assert_eq!(decode("a=b="), Err(Base64Error::Padding));
        assert_eq!(decode("!!!!"), Err(Base64Error::Char));
    }
}
