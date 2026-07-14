//! Policy-driven placement across storage resources and failure domains.

use std::collections::BTreeSet;

use flakecache_cas::ContentId;

/// Physical storage capability. Logical buckets never expose this value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BackendKind {
    /// A local or mounted filesystem, including PVC and iSCSI mounts.
    Filesystem,
    /// An S3-compatible or other durable object store.
    ObjectStore,
    /// An independent archival or WORM-capable failure domain.
    Archive,
}

/// A named physical storage resource in a region and zone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageResource {
    id: String,
    region: String,
    zone: String,
    backend: BackendKind,
}

impl StorageResource {
    /// Create a resource. IDs must be stable across catalog refreshes.
    pub fn new(
        id: impl Into<String>,
        region: impl Into<String>,
        zone: impl Into<String>,
        backend: BackendKind,
    ) -> Self {
        Self {
            id: id.into(),
            region: region.into(),
            zone: zone.into(),
            backend,
        }
    }

    /// Stable resource identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Region failure domain.
    #[must_use]
    pub fn region(&self) -> &str {
        &self.region
    }

    /// Zone failure domain.
    #[must_use]
    pub fn zone(&self) -> &str {
        &self.zone
    }

    /// Backend capability.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }
}

/// Durability objectives supplied by the product/control plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceObjective {
    replicas: usize,
    min_regions: usize,
    min_zones: usize,
    require_archive: bool,
}

impl ServiceObjective {
    /// Require `replicas` distinct physical resources.
    #[must_use]
    pub const fn new(replicas: usize) -> Self {
        Self {
            replicas,
            min_regions: 1,
            min_zones: 1,
            require_archive: false,
        }
    }

    /// Require placement across at least this many regions.
    #[must_use]
    pub const fn with_min_regions(mut self, regions: usize) -> Self {
        self.min_regions = regions;
        self
    }

    /// Require placement across at least this many zones.
    #[must_use]
    pub const fn with_min_zones(mut self, zones: usize) -> Self {
        self.min_zones = zones;
        self
    }

    /// Require one selected resource to be an independent archive.
    #[must_use]
    pub const fn requiring_archive(mut self) -> Self {
        self.require_archive = true;
        self
    }
}

/// Deterministic placement result containing physical resources only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementPlan {
    resources: Vec<StorageResource>,
}

impl PlacementPlan {
    /// Selected resources, in deterministic preference order.
    #[must_use]
    pub fn resources(&self) -> &[StorageResource] {
        &self.resources
    }

    /// Stable selected resource IDs.
    #[must_use]
    pub fn resource_ids(&self) -> Vec<&str> {
        self.resources.iter().map(StorageResource::id).collect()
    }

    /// Number of selected region failure domains.
    #[must_use]
    pub fn distinct_regions(&self) -> usize {
        self.resources
            .iter()
            .map(StorageResource::region)
            .collect::<BTreeSet<_>>()
            .len()
    }

    /// Number of selected zone failure domains.
    #[must_use]
    pub fn distinct_zones(&self) -> usize {
        self.resources
            .iter()
            .map(|resource| (resource.region(), resource.zone()))
            .collect::<BTreeSet<_>>()
            .len()
    }
}

/// An objective cannot be satisfied by the available catalog.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "placement objective cannot be satisfied: {required_replicas} replicas requested from {available_resources} resources"
)]
pub struct PolicyError {
    required_replicas: usize,
    available_resources: usize,
}

impl PolicyError {
    /// Requested physical replica count.
    #[must_use]
    pub const fn required_replicas(&self) -> usize {
        self.required_replicas
    }

    /// Number of available physical resources.
    #[must_use]
    pub const fn available_resources(&self) -> usize {
        self.available_resources
    }
}

/// A deterministic catalog-to-placement compiler.
#[derive(Debug, Clone, Default)]
pub struct PlacementPolicy {
    resources: Vec<StorageResource>,
}

impl PlacementPolicy {
    /// Build a catalog. Duplicate IDs resolve deterministically to the first
    /// resource after sorting by the full resource identity.
    pub fn new(resources: impl IntoIterator<Item = StorageResource>) -> Self {
        let mut resources: Vec<_> = resources.into_iter().collect();
        resources.sort_by(|left, right| {
            left.id
                .cmp(&right.id)
                .then_with(|| left.region.cmp(&right.region))
                .then_with(|| left.zone.cmp(&right.zone))
                .then_with(|| left.backend.cmp(&right.backend))
        });
        resources.dedup_by(|left, right| left.id == right.id);
        Self { resources }
    }

    /// Compile a content ID and service objective into physical placement.
    ///
    /// The compiler rejects unsatisfiable objectives. It never silently lowers
    /// durability to match the current catalog.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError`] if the catalog lacks the requested number of
    /// resources, regions, zones, or an independent archive.
    pub fn compile(
        &self,
        content: ContentId,
        objective: &ServiceObjective,
    ) -> Result<PlacementPlan, PolicyError> {
        if !self.can_satisfy(objective) {
            return Err(self.error(objective));
        }

        let mut ranked = self.resources.clone();
        ranked.sort_by(|left, right| {
            resource_weight(right, content)
                .cmp(&resource_weight(left, content))
                .then_with(|| left.id.cmp(&right.id))
        });

        let mut selected = Vec::with_capacity(objective.replicas);
        if objective.require_archive {
            let Some(archive) = ranked
                .iter()
                .find(|resource| resource.backend == BackendKind::Archive)
            else {
                return Err(self.error(objective));
            };
            selected.push(archive.clone());
        }

        while distinct_regions(&selected) < objective.min_regions {
            let regions = selected
                .iter()
                .map(StorageResource::region)
                .collect::<BTreeSet<_>>();
            let Some(resource) = ranked.iter().find(|candidate| {
                !contains_id(&selected, candidate.id()) && !regions.contains(candidate.region())
            }) else {
                return Err(self.error(objective));
            };
            selected.push(resource.clone());
        }

        while distinct_zones(&selected) < objective.min_zones {
            let zones = selected
                .iter()
                .map(|resource| (resource.region(), resource.zone()))
                .collect::<BTreeSet<_>>();
            let Some(resource) = ranked.iter().find(|candidate| {
                !contains_id(&selected, candidate.id())
                    && !zones.contains(&(candidate.region(), candidate.zone()))
            }) else {
                return Err(self.error(objective));
            };
            selected.push(resource.clone());
        }

        for resource in ranked {
            if selected.len() == objective.replicas {
                break;
            }
            if !contains_id(&selected, resource.id()) {
                selected.push(resource);
            }
        }

        if selected.len() != objective.replicas {
            return Err(self.error(objective));
        }
        Ok(PlacementPlan {
            resources: selected,
        })
    }

    fn can_satisfy(&self, objective: &ServiceObjective) -> bool {
        objective.replicas > 0
            && objective.replicas <= self.resources.len()
            && objective.min_regions > 0
            && objective.min_regions <= objective.replicas
            && objective.min_regions <= distinct_regions(&self.resources)
            && objective.min_zones > 0
            && objective.min_zones <= objective.replicas
            && objective.min_zones <= distinct_zones(&self.resources)
            && (!objective.require_archive
                || self
                    .resources
                    .iter()
                    .any(|resource| resource.backend == BackendKind::Archive))
    }

    fn error(&self, objective: &ServiceObjective) -> PolicyError {
        PolicyError {
            required_replicas: objective.replicas,
            available_resources: self.resources.len(),
        }
    }
}

fn contains_id(resources: &[StorageResource], id: &str) -> bool {
    resources.iter().any(|resource| resource.id == id)
}

fn distinct_regions(resources: &[StorageResource]) -> usize {
    resources
        .iter()
        .map(StorageResource::region)
        .collect::<BTreeSet<_>>()
        .len()
}

fn distinct_zones(resources: &[StorageResource]) -> usize {
    resources
        .iter()
        .map(|resource| (resource.region(), resource.zone()))
        .collect::<BTreeSet<_>>()
        .len()
}

fn resource_weight(resource: &StorageResource, content: ContentId) -> u64 {
    let mut input = Vec::with_capacity(resource.id.len() + content.as_bytes().len());
    input.extend_from_slice(resource.id.as_bytes());
    input.extend_from_slice(content.as_bytes());
    let digest = flakecache_crypto::shake256_256(&input);
    let mut leading = [0_u8; 8];
    leading.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(leading)
}
