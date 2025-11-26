use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use rustler::{Binary, Error, NifResult};

/// Generate a new Ed25519 keypair
/// Returns: {:ok, {`secret_key_base64`, `public_key_base64`}}
#[rustler::nif]
#[allow(clippy::unnecessary_wraps)] // NIFs require Result wrapper for error handling
pub fn generate_keypair() -> NifResult<(String, String)> {
    let mut csprng = OsRng;
    let signing_key = SigningKey::generate(&mut csprng);
    let verifying_key = signing_key.verifying_key();

    let secret_b64 = BASE64.encode(signing_key.as_bytes());
    let public_b64 = BASE64.encode(verifying_key.as_bytes());

    Ok((secret_b64, public_b64))
}

/// Sign data with Ed25519 secret key
/// Args: data (binary), `secret_key` (base64 string)
/// Returns: {:ok, `signature_base64`} or {:error, reason}
#[rustler::nif]
pub fn sign_data(data: Binary, secret_key_b64: &str) -> NifResult<String> {
    // Decode secret key with strict base64 validation
    let secret_bytes = BASE64.decode(secret_key_b64).map_err(|_| Error::BadArg)?;

    if secret_bytes.len() != 32 {
        return Err(Error::BadArg);
    }

    let mut key_bytes = [0u8; 32];
    key_bytes.copy_from_slice(&secret_bytes);

    let signing_key = SigningKey::from_bytes(&key_bytes);
    let signature = signing_key.sign(data.as_slice());

    Ok(BASE64.encode(signature.to_bytes().as_ref()))
}

/// Verify Ed25519 signature
/// Args: data (binary), signature (base64), `public_key` (base64)
/// Returns: :ok or {:error, `:invalid_signature`}
#[rustler::nif]
pub fn verify_signature(data: Binary, signature_b64: &str, public_key_b64: &str) -> NifResult<()> {
    // Decode public key with strict base64 validation
    let public_bytes = BASE64.decode(public_key_b64).map_err(|_| Error::BadArg)?;

    if public_bytes.len() != 32 {
        return Err(Error::BadArg);
    }

    let mut key_bytes = [0u8; 32];
    key_bytes.copy_from_slice(&public_bytes);

    let verifying_key = VerifyingKey::from_bytes(&key_bytes).map_err(|_| Error::BadArg)?;

    // Decode signature with strict base64 validation
    let sig_bytes = BASE64.decode(signature_b64).map_err(|_| Error::BadArg)?;

    if sig_bytes.len() != 64 {
        return Err(Error::BadArg);
    }

    let mut signature_bytes = [0u8; 64];
    signature_bytes.copy_from_slice(&sig_bytes);

    let signature = Signature::from_bytes(&signature_bytes);

    // Verify
    verifying_key
        .verify(data.as_slice(), &signature)
        .map_err(|_| Error::RaiseTerm(Box::new("invalid_signature")))?;

    Ok(())
}
