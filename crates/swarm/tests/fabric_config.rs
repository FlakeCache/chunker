use std::fs;
use std::path::Path;

use flakecache_cas::{ContentId, ObjectKind};
use flakecache_swarm::FabricConfig;

fn write_config(contents: &str) -> tempfile::NamedTempFile {
    let file = tempfile::NamedTempFile::new().unwrap();
    fs::write(file.path(), contents).unwrap();
    file
}

#[test]
fn yaml_catalog_drives_named_service_classes() {
    let file = write_config(
        r"
version: 1
resources:
  - id: local-eu-a
    region: eu
    zone: eu-a
    backend: filesystem
  - id: garage-eu-b
    region: eu
    zone: eu-b
    backend: object_store
  - id: garage-us-a
    region: us
    zone: us-a
    backend: object_store
  - id: archive-independent
    region: archive
    zone: archive-a
    backend: archive
service_classes:
  cheap:
    replicas: 1
    min_regions: 1
    min_zones: 1
  enterprise:
    replicas: 3
    min_regions: 2
    min_zones: 3
  immutable-release:
    replicas: 3
    min_regions: 2
    min_zones: 2
    require_archive: true
",
    );

    let config = FabricConfig::from_yaml_file(file.path()).unwrap();
    let content = ContentId::compute(ObjectKind::Chunk, b"artifact");
    let objective = config.service_class("enterprise").unwrap();
    let plan = config.placement().compile(content, objective).unwrap();

    assert_eq!(config.version(), 1);
    assert_eq!(plan.resources().len(), 3);
    assert!(plan.distinct_regions() >= 2);
    assert!(plan.distinct_zones() >= 3);
    assert!(config.service_class("not-configured").is_none());
}

#[test]
fn unsupported_schema_version_fails_loudly() {
    let file = write_config(
        r"
version: 2
resources: []
service_classes: {}
",
    );

    let error = FabricConfig::from_yaml_file(file.path()).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("unsupported fabric config version 2")
    );
}

#[test]
fn malformed_resource_is_rejected_at_load_time() {
    let file = write_config(
        r"
version: 1
resources:
  - id: missing-topology
    backend: filesystem
service_classes: {}
",
    );

    let error = FabricConfig::from_yaml_file(file.path()).unwrap_err();

    assert!(error.to_string().contains("region"));
}

#[test]
fn required_top_level_catalog_fields_must_be_present() {
    let missing_resources = write_config(
        r"
version: 1
service_classes: {}
",
    );
    let missing_classes = write_config(
        r"
version: 1
resources: []
",
    );

    assert!(
        FabricConfig::from_yaml_file(missing_resources.path())
            .unwrap_err()
            .to_string()
            .contains("resources")
    );
    assert!(
        FabricConfig::from_yaml_file(missing_classes.path())
            .unwrap_err()
            .to_string()
            .contains("service_classes")
    );
}

#[test]
fn repository_example_is_loadable_and_satisfiable() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../deploy/fabric.example.yaml");

    let config = FabricConfig::from_yaml_file(path).unwrap();

    assert!(config.service_class("self-hosted").is_some());
    assert!(config.service_class("multi-region").is_some());
    assert!(config.service_class("immutable-release").is_some());
}
