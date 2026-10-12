from dataclasses import replace

import pytest

from audioscope.manifest import read_manifest, validate_manifest
from audioscope.synthetic import generate_smoke_dataset


def test_generated_manifest_is_valid(tmp_path):
    manifest = generate_smoke_dataset(tmp_path, sample_rate=8000, duration_s=0.1, seed=1)
    result = validate_manifest(read_manifest(manifest), tmp_path)
    assert result["records"] == 30
    assert result["classes"] == 3
    assert result["folds"] == [1, 2, 3, 4, 5]


def test_source_leakage_is_rejected(tmp_path):
    manifest = generate_smoke_dataset(tmp_path, sample_rate=8000, duration_s=0.1, seed=1)
    records = read_manifest(manifest)
    records[1] = replace(records[1], source_recording_id=records[0].source_recording_id, fold=2)
    with pytest.raises(ValueError, match="source leakage"):
        validate_manifest(records, tmp_path, verify_hash=False)
    result = validate_manifest(records, tmp_path, verify_hash=False, allow_source_fold_overlap=True)
    assert result["source_fold_overlaps"] == [records[0].source_recording_id]
