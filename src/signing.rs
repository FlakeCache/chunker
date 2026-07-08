use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use zeroize::Zeroize;

#[derive(Debug, thiserror::Error, Clone, Copy)]
pub enum SigningError {
    #[error("invalid_secret_key: expected 32 bytes")]
    InvalidSecretKey,
    #[error("invalid_public_key: expected 32 bytes")]
    InvalidPublicKey,
    #[error("invalid_signature: expected 64 bytes")]
    InvalidSignature,
    #[error("decode_error")]
    DecodeError,
    #[error("verification_failed")]
    VerificationFailed,
}

/// Generate a new Ed25519 keypair
/// Returns: {:ok, {`secret_key_base64`, `public_key_base64`}}
#[must_use]
pub fn generate_keypair() -> (String, String) {
    let mut csprng = OsRng;
    let signing_key = SigningKey::generate(&mut csprng);
    let verifying_key = signing_key.verifying_key();

    let secret_b64 = BASE64.encode(signing_key.as_bytes());
    let public_b64 = BASE64.encode(verifying_key.as_bytes());

    (secret_b64, public_b64)
}

/// Sign data with Ed25519 secret key
/// Args: data (binary), `secret_key` (base64 string)
/// Returns: {:ok, `signature_base64`} or {:error, reason}
///
/// # Errors
///
/// Returns `SigningError` if the secret key is invalid or decoding fails.
pub fn sign_data(data: &[u8], secret_key_b64: &str) -> Result<String, SigningError> {
    // Decode secret key with strict base64 validation
    let mut secret_bytes = BASE64
        .decode(secret_key_b64)
        .map_err(|_| SigningError::DecodeError)?;

    if secret_bytes.len() != 32 {
        secret_bytes.zeroize();
        return Err(SigningError::InvalidSecretKey);
    }

    let mut key_bytes = [0u8; 32];
    key_bytes.copy_from_slice(&secret_bytes);
    // Zeroize the heap-allocated vector immediately after copying
    secret_bytes.zeroize();

    let signing_key = SigningKey::from_bytes(&key_bytes);
    // Zeroize the stack-allocated array
    key_bytes.zeroize();

    let signature = signing_key.sign(data);

    Ok(BASE64.encode(signature.to_bytes().as_ref()))
}

/// Verify Ed25519 signature
/// Args: data (binary), signature (base64), `public_key` (base64)
/// Returns: :ok or {:error, `:invalid_signature`}
///
/// # Errors
///
/// Returns `SigningError` if the signature or public key is invalid, or if verification fails.
pub fn verify_signature(
    data: &[u8],
    signature_b64: &str,
    public_key_b64: &str,
) -> Result<(), SigningError> {
    // Decode public key with strict base64 validation
    let public_bytes = BASE64
        .decode(public_key_b64)
        .map_err(|_| SigningError::DecodeError)?;

    if public_bytes.len() != 32 {
        return Err(SigningError::InvalidPublicKey);
    }

    let mut key_bytes = [0u8; 32];
    key_bytes.copy_from_slice(&public_bytes);

    let verifying_key =
        VerifyingKey::from_bytes(&key_bytes).map_err(|_| SigningError::InvalidPublicKey)?;

    // Decode signature with strict base64 validation
    let sig_bytes = BASE64
        .decode(signature_b64)
        .map_err(|_| SigningError::DecodeError)?;

    if sig_bytes.len() != 64 {
        return Err(SigningError::InvalidSignature);
    }

    let mut signature_bytes = [0u8; 64];
    signature_bytes.copy_from_slice(&sig_bytes);

    let signature = Signature::from_bytes(&signature_bytes);

    // Verify
    verifying_key
        .verify(data, &signature)
        .map_err(|_| SigningError::VerificationFailed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signing_errors() -> Result<(), SigningError> {
        let (secret, public) = generate_keypair();
        let data = b"test";
        let signature = sign_data(data, &secret)?;

        // Invalid secret key (base64 decode error)
        assert!(matches!(
            sign_data(data, "invalid-base64"),
            Err(SigningError::DecodeError)
        ));

        // Invalid secret key (wrong length)
        let short_key = BASE64.encode([0u8; 31]);
        assert!(matches!(
            sign_data(data, &short_key),
            Err(SigningError::InvalidSecretKey)
        ));

        // Invalid public key (base64 decode error)
        assert!(matches!(
            verify_signature(data, &signature, "invalid-base64"),
            Err(SigningError::DecodeError)
        ));

        // Invalid public key (wrong length)
        let short_pub = BASE64.encode([0u8; 31]);
        assert!(matches!(
            verify_signature(data, &signature, &short_pub),
            Err(SigningError::InvalidPublicKey)
        ));

        // Invalid signature (base64 decode error)
        assert!(matches!(
            verify_signature(data, "invalid-base64", &public),
            Err(SigningError::DecodeError)
        ));

        // Invalid signature (wrong length)
        let short_sig = BASE64.encode([0u8; 63]);
        assert!(matches!(
            verify_signature(data, &short_sig, &public),
            Err(SigningError::InvalidSignature)
        ));

        // Verification failed
        let other_sig = sign_data(b"other", &secret)?;
        assert!(matches!(
            verify_signature(data, &other_sig, &public),
            Err(SigningError::VerificationFailed)
        ));
        Ok(())
    }
}
