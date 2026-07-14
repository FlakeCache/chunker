use flakecache_cas::{ContentId, ObjectKind};
use flakecache_swarm::{BackendKind, PlacementPolicy, ServiceObjective, StorageResource};

fn key(seed: &[u8]) -> ContentId {
    ContentId::compute(ObjectKind::Chunk, seed)
}

fn resource(id: &str, region: &str, zone: &str, backend: BackendKind) -> StorageResource {
    StorageResource::new(id, region, zone, backend)
}

#[test]
fn standard_object_spans_zones_without_leaking_backend_details() {
    let policy = PlacementPolicy::new([
        resource("node-a", "eu", "eu-a", BackendKind::Filesystem),
        resource("node-b", "eu", "eu-b", BackendKind::Filesystem),
        resource("garage-eu", "eu", "eu-c", BackendKind::ObjectStore),
    ]);
    let objective = ServiceObjective::new(3).with_min_zones(2);

    let plan = policy.compile(key(b"standard"), &objective).unwrap();

    assert_eq!(plan.resources().len(), 3);
    assert!(plan.distinct_zones() >= 2);
    assert_eq!(plan.distinct_regions(), 1);
}

#[test]
fn premium_object_spans_regions() {
    let policy = PlacementPolicy::new([
        resource("eu-a", "eu", "eu-a", BackendKind::Filesystem),
        resource("eu-b", "eu", "eu-b", BackendKind::ObjectStore),
        resource("us-a", "us", "us-a", BackendKind::ObjectStore),
        resource("us-b", "us", "us-b", BackendKind::Filesystem),
    ]);
    let objective = ServiceObjective::new(3)
        .with_min_zones(3)
        .with_min_regions(2);

    let plan = policy.compile(key(b"premium"), &objective).unwrap();

    assert_eq!(plan.resources().len(), 3);
    assert!(plan.distinct_regions() >= 2);
    assert!(plan.distinct_zones() >= 3);
}

#[test]
fn immutable_object_requires_an_independent_archive() {
    let policy = PlacementPolicy::new([
        resource("eu", "eu", "eu-a", BackendKind::ObjectStore),
        resource("us", "us", "us-a", BackendKind::ObjectStore),
        resource("archive", "archive", "archive-a", BackendKind::Archive),
    ]);
    let objective = ServiceObjective::new(3)
        .with_min_regions(2)
        .requiring_archive();

    let plan = policy.compile(key(b"release"), &objective).unwrap();

    assert!(
        plan.resources()
            .iter()
            .any(|resource| resource.backend() == BackendKind::Archive)
    );
}

#[test]
fn unsatisfiable_objective_is_rejected_instead_of_weakened() {
    let policy = PlacementPolicy::new([
        resource("a", "eu", "eu-a", BackendKind::Filesystem),
        resource("b", "eu", "eu-b", BackendKind::ObjectStore),
    ]);
    let objective = ServiceObjective::new(3).with_min_regions(2);

    let error = policy.compile(key(b"impossible"), &objective).unwrap_err();

    assert_eq!(error.required_replicas(), 3);
    assert_eq!(error.available_resources(), 2);
}

#[test]
fn placement_is_deterministic_across_catalog_order() {
    let resources = [
        resource("a", "eu", "eu-a", BackendKind::Filesystem),
        resource("b", "eu", "eu-b", BackendKind::ObjectStore),
        resource("c", "us", "us-a", BackendKind::Filesystem),
        resource("d", "us", "us-b", BackendKind::ObjectStore),
    ];
    let reversed = resources.clone().into_iter().rev();
    let objective = ServiceObjective::new(3).with_min_regions(2);

    let first = PlacementPolicy::new(resources)
        .compile(key(b"same"), &objective)
        .unwrap();
    let second = PlacementPolicy::new(reversed)
        .compile(key(b"same"), &objective)
        .unwrap();

    assert_eq!(first.resource_ids(), second.resource_ids());
}
