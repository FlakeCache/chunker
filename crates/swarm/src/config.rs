//! Versioned on-disk fabric catalog and service-objective configuration.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use flakecache_cas::{ContentId, ObjectKind};
use serde::Deserialize;

use crate::{BackendKind, PlacementPolicy, ServiceObjective, StorageResource};

const SUPPORTED_VERSION: u32 = 1;

/// A loaded fabric resource catalog and named service objectives.
#[derive(Debug, Clone)]
pub struct FabricConfig {
    version: u32,
    placement: PlacementPolicy,
    service_classes: BTreeMap<String, ServiceObjective>,
}

impl FabricConfig {
    /// Load and validate a versioned YAML configuration from disk.
    ///
    /// This disk contract is deliberately independent of placement execution;
    /// a later CNPG catalog supplies the same typed records.
    ///
    /// # Errors
    ///
    /// Returns [`FabricConfigError`] when the file cannot be read, YAML does
    /// not match the schema, or the catalog cannot satisfy a configured class.
    pub fn from_yaml_file(path: impl AsRef<Path>) -> Result<Self, FabricConfigError> {
        let source = fs::read_to_string(path)?;
        let raw: RawConfig = serde_yaml::from_str(&source)?;
        Self::from_raw(raw)
    }

    /// Configuration schema version.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Validated physical placement catalog.
    #[must_use]
    pub const fn placement(&self) -> &PlacementPolicy {
        &self.placement
    }

    /// Named service objective, or `None` when the product has not configured it.
    #[must_use]
    pub fn service_class(&self, name: &str) -> Option<&ServiceObjective> {
        self.service_classes.get(name)
    }

    fn from_raw(raw: RawConfig) -> Result<Self, FabricConfigError> {
        if raw.version != SUPPORTED_VERSION {
            return Err(FabricConfigError::UnsupportedVersion(raw.version));
        }

        let mut ids = BTreeSet::new();
        let mut resources = Vec::with_capacity(raw.resources.len());
        for resource in raw.resources {
            validate_name("resource id", &resource.id)?;
            validate_name("region", &resource.region)?;
            validate_name("zone", &resource.zone)?;
            if !ids.insert(resource.id.clone()) {
                return Err(FabricConfigError::DuplicateResource(resource.id));
            }
            resources.push(StorageResource::new(
                resource.id,
                resource.region,
                resource.zone,
                resource.backend.into(),
            ));
        }
        let placement = PlacementPolicy::new(resources);

        let mut service_classes = BTreeMap::new();
        for (name, raw_objective) in raw.service_classes {
            validate_name("service class", &name)?;
            let mut objective = ServiceObjective::new(raw_objective.replicas)
                .with_min_regions(raw_objective.min_regions)
                .with_min_zones(raw_objective.min_zones);
            if raw_objective.require_archive {
                objective = objective.requiring_archive();
            }
            let validation_key = ContentId::compute(ObjectKind::Chunk, name.as_bytes());
            placement
                .compile(validation_key, &objective)
                .map_err(|source| FabricConfigError::UnsatisfiableServiceClass {
                    name: name.clone(),
                    source,
                })?;
            service_classes.insert(name, objective);
        }

        Ok(Self {
            version: raw.version,
            placement,
            service_classes,
        })
    }
}

fn validate_name(field: &'static str, value: &str) -> Result<(), FabricConfigError> {
    if value.trim().is_empty() {
        Err(FabricConfigError::EmptyField(field))
    } else {
        Ok(())
    }
}

/// Fabric configuration load or validation failure.
#[derive(Debug, thiserror::Error)]
pub enum FabricConfigError {
    /// The YAML file could not be read.
    #[error("read fabric config: {0}")]
    Read(#[from] std::io::Error),
    /// YAML did not conform to the schema.
    #[error("parse fabric config: {0}")]
    Parse(#[from] serde_yaml::Error),
    /// The loader refuses unknown schema semantics.
    #[error("unsupported fabric config version {0}")]
    UnsupportedVersion(u32),
    /// Required identity or topology is empty.
    #[error("{0} must not be empty")]
    EmptyField(&'static str),
    /// Resource identity must be globally unambiguous in one catalog.
    #[error("duplicate storage resource id {0}")]
    DuplicateResource(String),
    /// A named objective cannot be delivered by this resource catalog.
    #[error("service class {name} is unsatisfiable: {source}")]
    UnsatisfiableServiceClass {
        /// Configured service-class name.
        name: String,
        /// Placement validation failure.
        source: crate::PolicyError,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    version: u32,
    resources: Vec<RawResource>,
    service_classes: BTreeMap<String, RawObjective>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawResource {
    id: String,
    region: String,
    zone: String,
    backend: RawBackendKind,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RawBackendKind {
    Filesystem,
    ObjectStore,
    Archive,
}

impl From<RawBackendKind> for BackendKind {
    fn from(value: RawBackendKind) -> Self {
        match value {
            RawBackendKind::Filesystem => Self::Filesystem,
            RawBackendKind::ObjectStore => Self::ObjectStore,
            RawBackendKind::Archive => Self::Archive,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawObjective {
    replicas: usize,
    #[serde(default = "one")]
    min_regions: usize,
    #[serde(default = "one")]
    min_zones: usize,
    #[serde(default)]
    require_archive: bool,
}

const fn one() -> usize {
    1
}
