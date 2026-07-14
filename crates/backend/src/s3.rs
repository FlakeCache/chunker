// SPDX-License-Identifier: MIT
//! Path-style S3 cold storage with a small synchronous rustls client.

use std::env;
use std::io;
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use flakecache_cas::{BlobBackend, CasError, ContentId};
use sha2::{Digest, Sha256};

const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// Explicit configuration for a path-style S3-compatible cold tier.
#[derive(Clone)]
pub struct S3Config {
    /// S3 endpoint, for example `https://s3-eu1.centralcloud.com`.
    pub endpoint: String,
    /// Bucket containing content-addressed objects.
    pub bucket: String,
    /// `SigV4` region.
    pub region: String,
    /// `SigV4` access key id.
    pub access_key_id: String,
    /// `SigV4` secret access key.
    pub secret_access_key: String,
    /// Optional key prefix within the bucket.
    pub prefix: Option<String>,
}

impl std::fmt::Debug for S3Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3Config")
            .field("endpoint", &self.endpoint)
            .field("bucket", &self.bucket)
            .field("region", &self.region)
            .field("access_key_id", &"[redacted]")
            .field("secret_access_key", &"[redacted]")
            .field("prefix", &self.prefix)
            .finish()
    }
}

impl S3Config {
    /// Load configuration from `FLAKECACHE_S3_ENDPOINT`, `FLAKECACHE_S3_BUCKET`,
    /// `FLAKECACHE_S3_REGION`, `FLAKECACHE_S3_ACCESS_KEY_ID`,
    /// `FLAKECACHE_S3_SECRET_ACCESS_KEY`, and optional `FLAKECACHE_S3_PREFIX`.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when a required variable is absent.
    pub fn from_env() -> io::Result<Self> {
        fn required(name: &str) -> io::Result<String> {
            env::var(name).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("required environment variable {name} is not set"),
                )
            })
        }

        Ok(Self {
            endpoint: required("FLAKECACHE_S3_ENDPOINT")?,
            bucket: required("FLAKECACHE_S3_BUCKET")?,
            region: required("FLAKECACHE_S3_REGION")?,
            access_key_id: required("FLAKECACHE_S3_ACCESS_KEY_ID")?,
            secret_access_key: required("FLAKECACHE_S3_SECRET_ACCESS_KEY")?,
            prefix: env::var("FLAKECACHE_S3_PREFIX")
                .ok()
                .filter(|value| !value.is_empty()),
        })
    }
}

/// Durable S3-compatible cold backend using path-style requests and AWS `SigV4`.
///
/// Objects use `<prefix>/<first-two-content-id-hex>/<remaining-hex>`. The
/// synchronous client is intended for the node's existing synchronous
/// [`BlobBackend`] contract. `put` acknowledges only after S3 accepts the body.
#[derive(Clone)]
pub struct S3Backend {
    config: S3Config,
    agent: ureq::Agent,
    host: String,
    endpoint: String,
    endpoint_path: String,
}

impl std::fmt::Debug for S3Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3Backend")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl S3Backend {
    /// Construct a backend from explicit parameters.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] for a malformed endpoint or empty
    /// bucket, region, or credentials.
    pub fn new(config: S3Config) -> io::Result<Self> {
        let endpoint = config.endpoint.trim_end_matches('/').to_owned();
        let (_, remainder) = endpoint.split_once("://").ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "S3 endpoint needs http(s) scheme",
            )
        })?;
        if !(endpoint.starts_with("https://") || endpoint.starts_with("http://")) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "S3 endpoint needs http(s) scheme",
            ));
        }
        let (host, endpoint_path) = remainder
            .split_once('/')
            .map_or((remainder, String::new()), |(host, path)| {
                (host, format!("/{path}"))
            });
        if host.is_empty()
            || config.bucket.is_empty()
            || config.region.is_empty()
            || config.access_key_id.is_empty()
            || config.secret_access_key.is_empty()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "S3 endpoint host, bucket, region, and credentials must be non-empty",
            ));
        }
        Ok(Self {
            config,
            agent: ureq::Agent::new_with_defaults(),
            host: host.to_owned(),
            endpoint,
            endpoint_path,
        })
    }

    /// Construct from the `FLAKECACHE_S3_*` environment variables.
    ///
    /// # Errors
    /// Returns an error for missing configuration or a malformed endpoint.
    pub fn from_env() -> io::Result<Self> {
        Self::new(S3Config::from_env()?)
    }

    /// Return the S3 object key for a content id.
    #[must_use]
    pub fn key(&self, id: ContentId) -> String {
        let hex = id.to_hex();
        let shard = format!("{}/{}", &hex[..2], &hex[2..]);
        self.config.prefix.as_ref().map_or(shard.clone(), |prefix| {
            let prefix = prefix.trim_matches('/');
            if prefix.is_empty() {
                shard
            } else {
                format!("{prefix}/{shard}")
            }
        })
    }

    /// Check whether S3 contains `id` using `HEAD`.
    ///
    /// # Errors
    /// Returns [`CasError::Io`] on transport, authentication, or server errors.
    pub fn exists(&self, id: ContentId) -> Result<bool, CasError> {
        let request = self.signed_request("HEAD", id, EMPTY_SHA256)?;
        match self
            .agent
            .head(&request.url)
            .header("x-amz-content-sha256", EMPTY_SHA256)
            .header("x-amz-date", &request.amz_date)
            .header("authorization", &request.authorization)
            .call()
        {
            Ok(_) => Ok(true),
            Err(ureq::Error::StatusCode(404)) => Ok(false),
            Err(error) => Err(http_error(error)),
        }
    }

    /// Delete `id` from S3. Deleting an absent object is successful per S3.
    ///
    /// # Errors
    /// Returns [`CasError::Io`] on transport, authentication, or server errors.
    pub fn delete(&self, id: ContentId) -> Result<(), CasError> {
        let request = self.signed_request("DELETE", id, EMPTY_SHA256)?;
        self.agent
            .delete(&request.url)
            .header("x-amz-content-sha256", EMPTY_SHA256)
            .header("x-amz-date", &request.amz_date)
            .header("authorization", &request.authorization)
            .call()
            .map(|_| ())
            .map_err(http_error)
    }

    fn signed_request(
        &self,
        method: &str,
        id: ContentId,
        payload_hash: &str,
    ) -> Result<SignedRequest, CasError> {
        let key = uri_encode_path(&self.key(id));
        let uri = format!(
            "{}/{}/{}",
            self.endpoint_path,
            uri_encode_segment(&self.config.bucket),
            key
        );
        let amz_date = amz_date(SystemTime::now())?;
        let signed = sign(
            method,
            &uri,
            &self.host,
            payload_hash,
            &amz_date,
            &self.config.region,
            "s3",
            &self.config.access_key_id,
            &self.config.secret_access_key,
            true,
        );
        Ok(SignedRequest {
            url: format!(
                "{}{}",
                self.endpoint,
                uri.strip_prefix(&self.endpoint_path).unwrap_or(&uri)
            ),
            amz_date,
            authorization: signed.authorization,
        })
    }
}

impl BlobBackend for S3Backend {
    fn put(&self, id: ContentId, bytes: &[u8]) -> Result<(), CasError> {
        let payload_hash = hex(&Sha256::digest(bytes));
        let request = self.signed_request("PUT", id, &payload_hash)?;
        self.agent
            .put(&request.url)
            .header("x-amz-content-sha256", &payload_hash)
            .header("x-amz-date", &request.amz_date)
            .header("authorization", &request.authorization)
            .send(bytes)
            .map(|_| ())
            .map_err(http_error)
    }

    fn get(&self, id: ContentId) -> Result<Option<Bytes>, CasError> {
        let request = self.signed_request("GET", id, EMPTY_SHA256)?;
        match self
            .agent
            .get(&request.url)
            .header("x-amz-content-sha256", EMPTY_SHA256)
            .header("x-amz-date", &request.amz_date)
            .header("authorization", &request.authorization)
            .call()
        {
            Ok(mut response) => response
                .body_mut()
                .read_to_vec()
                .map(Bytes::from)
                .map(Some)
                .map_err(http_error),
            Err(ureq::Error::StatusCode(404)) => Ok(None),
            Err(error) => Err(http_error(error)),
        }
    }
}

struct SignedRequest {
    url: String,
    amz_date: String,
    authorization: String,
}

struct Signature {
    authorization: String,
    #[cfg(test)]
    canonical_request: String,
    #[cfg(test)]
    string_to_sign: String,
}

#[allow(clippy::too_many_arguments)]
fn sign(
    method: &str,
    canonical_uri: &str,
    host: &str,
    payload_hash: &str,
    amz_date: &str,
    region: &str,
    service: &str,
    access_key_id: &str,
    secret_access_key: &str,
    include_content_hash: bool,
) -> Signature {
    let date = &amz_date[..8];
    let (canonical_headers, signed_headers) = if include_content_hash {
        (
            format!("host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n"),
            "host;x-amz-content-sha256;x-amz-date",
        )
    } else {
        (
            format!("host:{host}\nx-amz-date:{amz_date}\n"),
            "host;x-amz-date",
        )
    };
    let canonical_request = format!(
        "{method}\n{canonical_uri}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    );
    let scope = format!("{date}/{region}/{service}/aws4_request");
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        hex(&Sha256::digest(canonical_request.as_bytes()))
    );
    let date_key = hmac_sha256(
        format!("AWS4{secret_access_key}").as_bytes(),
        date.as_bytes(),
    );
    let region_key = hmac_sha256(&date_key, region.as_bytes());
    let service_key = hmac_sha256(&region_key, service.as_bytes());
    let signing_key = hmac_sha256(&service_key, b"aws4_request");
    let signature = hex(&hmac_sha256(&signing_key, string_to_sign.as_bytes()));
    Signature {
        authorization: format!(
            "AWS4-HMAC-SHA256 Credential={access_key_id}/{scope}, SignedHeaders={signed_headers}, Signature={signature}"
        ),
        #[cfg(test)]
        canonical_request,
        #[cfg(test)]
        string_to_sign,
    }
}

fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    const BLOCK: usize = 64;
    let mut normalized = [0_u8; BLOCK];
    if key.len() > BLOCK {
        normalized[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        normalized[..key.len()].copy_from_slice(key);
    }
    let mut inner_pad = [0x36_u8; BLOCK];
    let mut outer_pad = [0x5c_u8; BLOCK];
    for index in 0..BLOCK {
        inner_pad[index] ^= normalized[index];
        outer_pad[index] ^= normalized[index];
    }
    let inner = Sha256::new()
        .chain_update(inner_pad)
        .chain_update(message)
        .finalize();
    Sha256::new()
        .chain_update(outer_pad)
        .chain_update(inner)
        .finalize()
        .into()
}

fn uri_encode_path(value: &str) -> String {
    value
        .split('/')
        .map(uri_encode_segment)
        .collect::<Vec<_>>()
        .join("/")
}

fn uri_encode_segment(value: &str) -> String {
    let mut encoded = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push('%');
            encoded.push(char::from(b"0123456789ABCDEF"[(byte >> 4) as usize]));
            encoded.push(char::from(b"0123456789ABCDEF"[(byte & 0x0f) as usize]));
        }
    }
    encoded
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from(b"0123456789abcdef"[(byte >> 4) as usize]));
        out.push(char::from(b"0123456789abcdef"[(byte & 0x0f) as usize]));
    }
    out
}

fn amz_date(now: SystemTime) -> Result<String, CasError> {
    let seconds = now
        .duration_since(UNIX_EPOCH)
        .map_err(|error| CasError::Io(io::Error::other(error)))?
        .as_secs();
    let days =
        i64::try_from(seconds / 86_400).map_err(|error| CasError::Io(io::Error::other(error)))?;
    let day_seconds = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    Ok(format!(
        "{year:04}{month:02}{day:02}T{:02}{:02}{:02}Z",
        day_seconds / 3_600,
        (day_seconds % 3_600) / 60,
        day_seconds % 60
    ))
}

fn civil_from_days(days_since_epoch: i64) -> (i64, i64, i64) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month, day)
}

fn http_error(error: ureq::Error) -> CasError {
    CasError::Io(io::Error::other(error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use flakecache_cas::ObjectKind;

    #[test]
    fn sigv4_matches_aws_get_vanilla_test_suite_vector() {
        // Official AWS aws-sig-v4-test-suite/get-vanilla fixture, mirrored by
        // botocore under tests/unit/auth/aws4_testsuite/get-vanilla/.
        let signed = sign(
            "GET",
            "/",
            "example.amazonaws.com",
            EMPTY_SHA256,
            "20150830T123600Z",
            "us-east-1",
            "service",
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            false,
        );

        assert_eq!(
            signed.canonical_request,
            concat!(
                "GET\n/\n\n",
                "host:example.amazonaws.com\n",
                "x-amz-date:20150830T123600Z\n\n",
                "host;x-amz-date\n",
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            )
        );
        assert_eq!(
            signed.string_to_sign,
            concat!(
                "AWS4-HMAC-SHA256\n",
                "20150830T123600Z\n",
                "20150830/us-east-1/service/aws4_request\n",
                "bb579772317eb040ac9ed261061d46c1f17a8133879d6129b6e1c25292927e63"
            )
        );
        assert_eq!(
            signed.authorization,
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, SignedHeaders=host;x-amz-date, Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
        );
    }

    #[test]
    fn key_is_sharded_below_optional_prefix() {
        let backend = S3Backend::new(S3Config {
            endpoint: "https://s3-eu1.centralcloud.com".into(),
            bucket: "bucket".into(),
            region: "garage".into(),
            access_key_id: "access".into(),
            secret_access_key: "secret".into(),
            prefix: Some("fabric/chunks/".into()),
        })
        .unwrap();
        let id = ContentId::from_bytes([0xab; 32]);
        assert_eq!(
            backend.key(id),
            format!("fabric/chunks/ab/{}", "ab".repeat(31))
        );
    }

    #[test]
    #[ignore = "requires FLAKECACHE_S3_TEST=1 and live S3 credentials"]
    fn live_s3_round_trip_put_get_exists_delete() -> Result<(), Box<dyn std::error::Error>> {
        if env::var("FLAKECACHE_S3_TEST").as_deref() != Ok("1") {
            eprintln!("live S3 round trip skipped: FLAKECACHE_S3_TEST is not 1");
            return Ok(());
        }
        let backend = S3Backend::from_env()?;
        let payload = format!(
            "flakecache-s3-integration-{}",
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        );
        let id = ContentId::compute(ObjectKind::Chunk, payload.as_bytes());
        backend.put(id, payload.as_bytes())?;
        assert!(backend.exists(id)?);
        assert_eq!(backend.get(id)?.as_deref(), Some(payload.as_bytes()));
        backend.delete(id)?;
        assert!(!backend.exists(id)?);
        assert!(backend.get(id)?.is_none());
        Ok(())
    }
}
