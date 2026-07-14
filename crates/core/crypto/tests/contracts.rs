// SPDX-License-Identifier: MIT

#[cfg(feature = "ed25519")]
use ed25519_dalek::SigningKey;
use flakecache_crypto::{WitnessEntry, append_witness_entry, shake256_256, verify_witness_chain};
#[cfg(feature = "ed25519")]
use flakecache_crypto::{sign_message, verify_message};

#[cfg(feature = "ed25519")]
#[test]
fn sign_verify_round_trip() {
    let signing_key = SigningKey::from_bytes(&[7_u8; 32]);
    let verifying_key = signing_key.verifying_key();
    let message = b"flakecache manifest v1";

    let signature = sign_message(message, &signing_key);

    assert!(verify_message(message, &signature, &verifying_key));
    assert!(!verify_message(
        b"tampered manifest",
        &signature,
        &verifying_key
    ));
}

#[test]
fn witness_chain_append_and_verify() {
    let mut chain = Vec::new();
    let first = WitnessEntry::new(shake256_256(b"put chunk A"), 10, 1);
    let second = WitnessEntry::new(shake256_256(b"commit manifest"), 11, 2);

    append_witness_entry(&mut chain, &first).expect("append genesis witness");
    append_witness_entry(&mut chain, &second).expect("append second witness");

    let verified = verify_witness_chain(&chain).expect("verify appended witness chain");
    assert_eq!(verified.len(), 2);
    assert_eq!(verified[0].action_hash, first.action_hash);
    assert_eq!(verified[1].action_hash, second.action_hash);

    chain[80] ^= 0xff;
    assert!(verify_witness_chain(&chain).is_err());
}
