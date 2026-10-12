from audioscope.db import SCHEMA, Base


def test_catalog_metadata_contains_release_lineage_tables():
    expected = {
        "sources",
        "licenses",
        "datasets",
        "dataset_versions",
        "audio_files",
        "annotations",
        "dataset_splits",
        "processing_runs",
        "model_artifacts",
        "benchmark_results",
        "field_sessions",
        "field_recordings",
        "quality_checks",
        "release_audits",
    }
    observed = {table.name for table in Base.metadata.tables.values() if table.schema == SCHEMA}
    assert observed == expected


def test_audio_foreign_keys_use_catalog_schema():
    table = Base.metadata.tables[f"{SCHEMA}.audio_files"]
    targets = {key.target_fullname for key in table.foreign_keys}
    assert targets == {f"{SCHEMA}.dataset_versions.id"}
