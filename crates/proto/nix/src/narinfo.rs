// SPDX-License-Identifier: MIT
//! Nix narinfo rendering and Ed25519 signing.
//!
//! A Nix binary-cache client substitutes a store path only when the path's
//! `.narinfo` carries a `Sig:` line that verifies — against a key in the
//! client's `trusted-public-keys` — over Nix's canonical *fingerprint*:
//!
//! ```text
//! 1;<store-path>;sha256:<nar-hash>;<nar-size>;<comma-joined-references>
//! ```
//!
//! where `<nar-hash>` is the uncompressed NAR's SHA-256 digest in Nix's base-32
//! alphabet and each reference is a full store path. A single wrong byte in that
//! preimage produces a signature Nix silently rejects (falling back to a local
//! rebuild), so [`NarInfo::fingerprint`] and [`NarInfo::signature`] are pinned by
//! a known-answer test against real `nix store sign` output.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use flakecache_crypto::b64;
use flakecache_crypto::{
    SigningKey, VerifyingKey, sign_message, signing_key_from_seed, verify_message,
};

/// Nix's base-32 alphabet (RFC 4648 minus `e`, `o`, `t`, `u`).
const NIX_BASE32: &[u8; 32] = b"0123456789abcdfghijklmnpqrsvwxyz";

/// Encode a byte digest in Nix's base-32, as used for `sha256:` NAR hashes.
///
/// This is the little-endian, high-index-first scheme from Nix's `libutil`, not
/// RFC 4648 base32; a 32-byte digest yields 52 characters.
#[must_use]
pub fn nix_base32(digest: &[u8]) -> String {
    if digest.is_empty() {
        return String::new();
    }
    let len = (digest.len() * 8 - 1) / 5 + 1;
    let mut out = String::with_capacity(len);
    for n in (0..len).rev() {
        let b = n * 5;
        let i = b / 8;
        let j = b % 8;
        let mut c = u32::from(digest[i]) >> j;
        if i + 1 < digest.len() {
            c |= u32::from(digest[i + 1]) << (8 - j);
        }
        out.push(NIX_BASE32[(c & 0x1f) as usize] as char);
    }
    out
}

/// Errors from [`parse_secret_key`].
#[derive(Debug, PartialEq, Eq)]
pub enum KeyError {
    /// The key is not of the form `<name>:<base64>`.
    Format,
    /// The base64 body did not decode.
    Base64,
    /// The decoded key is shorter than the 32-byte Ed25519 seed.
    Length,
}

impl std::fmt::Display for KeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Format => f.write_str("nix key: expected <name>:<base64>"),
            Self::Base64 => f.write_str("nix key: invalid base64"),
            Self::Length => f.write_str("nix key: decoded key shorter than 32 bytes"),
        }
    }
}

impl std::error::Error for KeyError {}

/// Parse a Nix binary-cache secret key (`<name>:<base64-of-64-bytes>`).
///
/// The 64 bytes are the 32-byte Ed25519 seed followed by the 32-byte public
/// key; the returned [`SigningKey`] is built from the seed.
///
/// # Errors
/// Returns [`KeyError`] if the format, base64, or key length is invalid.
pub fn parse_secret_key(text: &str) -> Result<(String, SigningKey), KeyError> {
    let (name, body) = text.trim().split_once(':').ok_or(KeyError::Format)?;
    let raw = b64::decode(body).map_err(|_| KeyError::Base64)?;
    let seed: [u8; 32] = raw
        .get(..32)
        .and_then(|s| s.try_into().ok())
        .ok_or(KeyError::Length)?;
    Ok((name.to_string(), signing_key_from_seed(&seed)))
}

/// The last `/`-separated component of a store path (its basename).
fn basename(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// The fields of a narinfo required to render and sign it.
///
/// `nar_hash` and `file_hash` are raw 32-byte SHA-256 digests; they are rendered
/// as `sha256:<nix-base32>` in the narinfo body and (for `nar_hash`) the
/// fingerprint.
pub struct NarInfo {
    /// Full store path, e.g. `/nix/store/<hash>-<name>`.
    pub store_path: String,
    /// Relative URL of the (compressed) NAR, e.g. `nar/<file-hash>.nar.xz`.
    pub url: String,
    /// Compression algorithm: `xz`, `zstd`, or `none`.
    pub compression: String,
    /// SHA-256 of the (compressed) file served at `url`.
    pub file_hash: [u8; 32],
    /// Size in bytes of the file served at `url`.
    pub file_size: u64,
    /// SHA-256 of the uncompressed NAR.
    pub nar_hash: [u8; 32],
    /// Size in bytes of the uncompressed NAR.
    pub nar_size: u64,
    /// Full store paths this path references (may include itself).
    pub references: Vec<String>,
    /// Optional deriver store path.
    pub deriver: Option<String>,
}

impl NarInfo {
    /// Nix's canonical signing fingerprint for this path.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        format!(
            "1;{};sha256:{};{};{}",
            self.store_path,
            nix_base32(&self.nar_hash),
            self.nar_size,
            self.references.join(",")
        )
    }

    /// The `Sig:` value (`<key-name>:<base64-signature>`) over [`Self::fingerprint`].
    #[must_use]
    pub fn signature(&self, key_name: &str, secret_key: &SigningKey) -> String {
        let sig = sign_message(self.fingerprint().as_bytes(), secret_key);
        format!("{key_name}:{}", b64::encode(&sig))
    }

    /// Render the full narinfo body, appending one `Sig:` line per signature.
    ///
    /// References and the deriver are rendered as basenames, matching Nix's
    /// narinfo format (the store dir is implied by `StoreDir`).
    #[must_use]
    pub fn to_text(&self, signatures: &[String]) -> String {
        let mut s = String::new();
        // Infallible: writing into a String never errors.
        let _ = writeln!(s, "StorePath: {}", self.store_path);
        let _ = writeln!(s, "URL: {}", self.url);
        let _ = writeln!(s, "Compression: {}", self.compression);
        let _ = writeln!(s, "FileHash: sha256:{}", nix_base32(&self.file_hash));
        let _ = writeln!(s, "FileSize: {}", self.file_size);
        let _ = writeln!(s, "NarHash: sha256:{}", nix_base32(&self.nar_hash));
        let _ = writeln!(s, "NarSize: {}", self.nar_size);
        let refs = self
            .references
            .iter()
            .map(|r| basename(r))
            .collect::<Vec<_>>()
            .join(" ");
        let _ = writeln!(s, "References: {refs}");
        if let Some(deriver) = &self.deriver {
            let _ = writeln!(s, "Deriver: {}", basename(deriver));
        }
        for sig in signatures {
            let _ = writeln!(s, "Sig: {sig}");
        }
        s
    }
}

/// Errors from [`NarInfo::parse`].
#[derive(Debug, PartialEq, Eq)]
pub enum ParseError {
    /// A required field was absent.
    MissingField(&'static str),
    /// A hash field was not `sha256:<nix-base32>`.
    BadHash,
    /// A numeric field did not parse.
    BadNumber,
    /// A base-32 digest did not decode.
    Base32,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingField(name) => write!(f, "narinfo: missing field {name}"),
            Self::BadHash => f.write_str("narinfo: hash not sha256:<nix-base32>"),
            Self::BadNumber => f.write_str("narinfo: invalid number"),
            Self::Base32 => f.write_str("narinfo: invalid nix-base32"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Decode a Nix base-32 string into its byte digest (inverse of [`nix_base32`]).
///
/// # Errors
/// Returns [`ParseError::Base32`] on a character outside the alphabet or if the
/// top bits overflow the digest length.
pub fn nix_base32_decode(s: &str) -> Result<Vec<u8>, ParseError> {
    let hash_size = s.len() * 5 / 8;
    let mut hash = vec![0u8; hash_size];
    for (n, &c) in s.as_bytes().iter().rev().enumerate() {
        let digit = u16::try_from(
            NIX_BASE32
                .iter()
                .position(|&x| x == c)
                .ok_or(ParseError::Base32)?,
        )
        .map_err(|_| ParseError::Base32)?;
        let b = n * 5;
        let i = b / 8;
        let j = b % 8;
        let shifted = digit << j;
        // `shifted` holds at most 13 significant bits; the split into a low byte
        // (stored) and a carry byte (folded into the next digest byte) is exact.
        #[allow(clippy::cast_possible_truncation)]
        let low = (shifted & 0xff) as u8;
        #[allow(clippy::cast_possible_truncation)]
        let carry = (shifted >> 8) as u8;
        hash[i] |= low;
        if i + 1 < hash_size {
            hash[i + 1] |= carry;
        } else if carry != 0 {
            return Err(ParseError::Base32);
        }
    }
    Ok(hash)
}

/// Parse a `sha256:<nix-base32>` hash field into its 32-byte digest.
fn parse_sha256(value: &str) -> Result<[u8; 32], ParseError> {
    let b32 = value.strip_prefix("sha256:").ok_or(ParseError::BadHash)?;
    nix_base32_decode(b32)?
        .try_into()
        .map_err(|_| ParseError::BadHash)
}

/// Parse a Nix public key (`<name>:<base64-of-32-bytes>`).
///
/// # Errors
/// Returns [`KeyError`] if the format, base64, or key length is invalid.
pub fn parse_public_key(text: &str) -> Result<(String, VerifyingKey), KeyError> {
    let (name, body) = text.trim().split_once(':').ok_or(KeyError::Format)?;
    let raw = b64::decode(body).map_err(|_| KeyError::Base64)?;
    let arr: [u8; 32] = raw.as_slice().try_into().map_err(|_| KeyError::Length)?;
    let key = VerifyingKey::from_bytes(&arr).map_err(|_| KeyError::Length)?;
    Ok((name.to_string(), key))
}

impl NarInfo {
    /// Parse a narinfo body, expanding reference/deriver basenames to full store
    /// paths under `store_dir` (needed to reconstruct the signing fingerprint).
    ///
    /// Returns the parsed metadata and the list of `Sig:` values present.
    ///
    /// # Errors
    /// Returns [`ParseError`] on a missing or malformed field.
    pub fn parse(text: &str, store_dir: &str) -> Result<(Self, Vec<String>), ParseError> {
        let mut store_path = None;
        let mut url = None;
        let mut compression = None;
        let mut file_hash = None;
        let mut file_size = None;
        let mut nar_hash = None;
        let mut nar_size = None;
        let mut references = Vec::new();
        let mut deriver = None;
        let mut sigs = Vec::new();

        for line in text.lines() {
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let value = value.trim();
            match key {
                "StorePath" => store_path = Some(value.to_string()),
                "URL" => url = Some(value.to_string()),
                "Compression" => compression = Some(value.to_string()),
                "FileHash" => file_hash = Some(parse_sha256(value)?),
                "FileSize" => {
                    file_size = Some(value.parse().map_err(|_| ParseError::BadNumber)?);
                }
                "NarHash" => nar_hash = Some(parse_sha256(value)?),
                "NarSize" => {
                    nar_size = Some(value.parse().map_err(|_| ParseError::BadNumber)?);
                }
                "References" => {
                    references = value
                        .split_whitespace()
                        .map(|r| format!("{store_dir}/{r}"))
                        .collect();
                }
                "Deriver" if !value.is_empty() => deriver = Some(format!("{store_dir}/{value}")),
                "Sig" => sigs.push(value.to_string()),
                _ => {}
            }
        }

        let narinfo = Self {
            store_path: store_path.ok_or(ParseError::MissingField("StorePath"))?,
            url: url.ok_or(ParseError::MissingField("URL"))?,
            compression: compression.unwrap_or_else(|| "none".to_string()),
            file_hash: file_hash.unwrap_or([0u8; 32]),
            file_size: file_size.unwrap_or(0),
            nar_hash: nar_hash.ok_or(ParseError::MissingField("NarHash"))?,
            nar_size: nar_size.ok_or(ParseError::MissingField("NarSize"))?,
            references,
            deriver,
        };
        Ok((narinfo, sigs))
    }

    /// Whether any of `sigs` is a valid signature over this path's fingerprint
    /// under a key present in `trusted` (keyed by Nix key name).
    ///
    /// This is the ingest check for the zero-knowledge model: a client signs the
    /// narinfo, and the cache accepts it only if the signature verifies against
    /// the cache's registered public key.
    #[must_use]
    pub fn verify_any(&self, sigs: &[String], trusted: &BTreeMap<String, VerifyingKey>) -> bool {
        let fingerprint = self.fingerprint();
        sigs.iter().any(|sig| {
            let Some((name, b64sig)) = sig.split_once(':') else {
                return false;
            };
            let Some(key) = trusted.get(name) else {
                return false;
            };
            let Ok(raw) = b64::decode(b64sig) else {
                return false;
            };
            let Ok(bytes) = <[u8; 64]>::try_from(raw.as_slice()) else {
                return false;
            };
            verify_message(fingerprint.as_bytes(), &bytes, key)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Known answer generated with real `nix` on 2026-07-14:
    //   nix-store --generate-binary-cache-key fixture-key-1 sk pk
    //   SP=$(nix-store --add fixture-file.txt)   # "flakecache narinfo fixture known-answer"
    //   nix store sign --key-file sk "$SP"
    //   nix path-info --json --sigs "$SP"
    const SECRET: &str = "fixture-key-1:99bHJJ1CGMrMElMB/KqKFVUCmCC+/oMKIUqrbR5ANyZI9BxZD0RyEPCwAOy98VPUgaviuSqMm3NKGUNwRjevAw==";
    const PUBLIC_B64: &str = "SPQcWQ9EchDwsADsvfFT1IGr4rkqjJtzShlDcEY3rwM=";
    const NAR_HASH_SRI_B64: &str = "LHmbxrZuoUhZ7p0Q+oD4wBzxH/byKBAKbJsfAZXoBCU=";
    const STORE_PATH: &str = "/nix/store/i2s5m9mjak08545k748i1a67y3yi04h5-fixture-file.txt";
    const EXPECTED_NIXB32: &str = "0984x2ah27wvdh510a7jyqgz2760z20gl44xxrcli8bfnv39ny9c";
    const EXPECTED_SIG: &str = "fixture-key-1:8iuXYdZXlQDjYZLFnCBJIUHAIO+GNGJVuoukQD5068EqMEtcJjEvYPlK5MYUjnfKmAxIEOGb9o8RiBRTu1KzAg==";

    fn nar_hash() -> [u8; 32] {
        b64::decode(NAR_HASH_SRI_B64).unwrap().try_into().unwrap()
    }

    fn fixture() -> NarInfo {
        NarInfo {
            store_path: STORE_PATH.to_string(),
            url: "nar/placeholder.nar.xz".to_string(),
            compression: "xz".to_string(),
            file_hash: [0u8; 32],
            file_size: 0,
            nar_hash: nar_hash(),
            nar_size: 152,
            references: Vec::new(),
            deriver: None,
        }
    }

    #[test]
    fn nix_base32_matches_nix() {
        assert_eq!(nix_base32(&nar_hash()), EXPECTED_NIXB32);
    }

    #[test]
    fn fingerprint_matches_nix() {
        assert_eq!(
            fixture().fingerprint(),
            format!("1;{STORE_PATH};sha256:{EXPECTED_NIXB32};152;")
        );
    }

    #[test]
    fn signature_reproduces_real_nix_signature() {
        let (name, secret_key) = parse_secret_key(SECRET).unwrap();
        assert_eq!(name, "fixture-key-1");
        assert_eq!(fixture().signature(&name, &secret_key), EXPECTED_SIG);
    }

    #[test]
    fn parsed_seed_yields_the_advertised_public_key() {
        let (_name, secret_key) = parse_secret_key(SECRET).unwrap();
        let derived = b64::encode(secret_key.verifying_key().as_bytes());
        assert_eq!(
            derived, PUBLIC_B64,
            "seed must derive the fixture public key"
        );
    }

    #[test]
    fn signed_narinfo_body_carries_the_sig_line() {
        let (name, secret_key) = parse_secret_key(SECRET).unwrap();
        let ni = fixture();
        let sig = ni.signature(&name, &secret_key);
        let body = ni.to_text(std::slice::from_ref(&sig));
        assert!(body.contains(&format!("Sig: {EXPECTED_SIG}\n")));
        assert!(body.starts_with(&format!("StorePath: {STORE_PATH}\n")));
        assert!(body.contains("References: \n")); // empty refs
    }

    // A real cache.nixos.org narinfo WITH a reference — pins the fingerprint's
    // reference-joining (basename -> full path) and verification against the
    // actual NixOS cache public key, not just self-consistency.
    const CACHE_NIXOS_PUBKEY: &str =
        "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY=";
    const ZLIB_NARINFO: &str = "StorePath: /nix/store/00xpncfcvafhr6vx9q05hkhazm70zw5g-zlib-ng-2.2.5\n\
URL: nar/0mcwgr3fyxb7id1vzl80m5mbrqr32y7s6jqqn1ac7ijq93i4i47a.nar.xz\n\
Compression: xz\n\
FileHash: sha256:0mcwgr3fyxb7id1vzl80m5mbrqr32y7s6jqqn1ac7ijq93i4i47a\n\
FileSize: 70984\n\
NarHash: sha256:14wdgrnw88fshrs36jw910m303sz7ybv7x33v95c19d6x6acxmnc\n\
NarSize: 171768\n\
References: xx7cm72qy2c0643cm1ipngd87aqwkcdp-glibc-2.40-66\n\
Deriver: wxk60x12j7g8yha1kwc6z4rpnanf45i6-zlib-ng-2.2.5.drv\n\
Sig: cache.nixos.org-1:VPSNg+AsqePO6v0qE8SchyDLuLYtqgZT7aocUYO5fU8LUfQ0f6fygOLCLRy6DAnual5jiWatUqABQU+R0usgBg==\n";

    fn trusted(pubkey_line: &str) -> BTreeMap<String, VerifyingKey> {
        let (name, key) = parse_public_key(pubkey_line).unwrap();
        BTreeMap::from([(name, key)])
    }

    #[test]
    fn nix_base32_decode_inverts_encode() {
        assert_eq!(nix_base32_decode(EXPECTED_NIXB32).unwrap(), nar_hash());
        let re = nix_base32(&nix_base32_decode(EXPECTED_NIXB32).unwrap());
        assert_eq!(re, EXPECTED_NIXB32);
    }

    #[test]
    fn verifies_real_cache_nixos_org_signature() {
        let (ni, sigs) = NarInfo::parse(ZLIB_NARINFO, "/nix/store").unwrap();
        assert_eq!(
            ni.references,
            vec!["/nix/store/xx7cm72qy2c0643cm1ipngd87aqwkcdp-glibc-2.40-66".to_string()],
            "reference basename must expand to a full store path"
        );
        assert!(
            ni.verify_any(&sigs, &trusted(CACHE_NIXOS_PUBKEY)),
            "a real cache.nixos.org signature must verify"
        );
    }

    #[test]
    fn rejects_untrusted_name_and_tampered_fields() {
        let (mut ni, sigs) = NarInfo::parse(ZLIB_NARINFO, "/nix/store").unwrap();
        // The signature's key name is not in the trusted set.
        assert!(!ni.verify_any(&sigs, &trusted(&format!("fixture-key-1:{PUBLIC_B64}"))));
        // Tampering any signed field breaks verification against the right key.
        ni.nar_size += 1;
        assert!(!ni.verify_any(&sigs, &trusted(CACHE_NIXOS_PUBKEY)));
    }

    #[test]
    fn zero_knowledge_round_trip_client_signs_server_verifies() {
        // Client holds the secret, signs; cache parses the uploaded narinfo and
        // accepts it only if it verifies against the cache's registered pubkey.
        let (name, secret_key) = parse_secret_key(SECRET).unwrap();
        let sig = fixture().signature(&name, &secret_key);
        let body = fixture().to_text(std::slice::from_ref(&sig));
        let (parsed, sigs) = NarInfo::parse(&body, "/nix/store").unwrap();
        assert!(parsed.verify_any(&sigs, &trusted(&format!("{name}:{PUBLIC_B64}"))));
    }
}
