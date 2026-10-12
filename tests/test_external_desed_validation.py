"""Deterministic integrity/dedup helper regressions, independent of network/model."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "external_validation", Path(__file__).resolve().parents[1] / "scripts/validate_external_desed.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_valid_variable_duration_preserves_events():
    events = [{"onset_s": 0., "offset_s": 11.}]
    module.validate_events(events, 11.)
    assert events[0]["offset_s"] == 11.


@pytest.mark.parametrize("onset,offset", [(-.1, 1.), (1., 1.), (0., 10.01), (0., float("nan")), (0., float("inf"))])
def test_bad_boundary_rejected(onset, offset):
    with pytest.raises(ValueError):
        module.validate_events([{"onset_s": onset, "offset_s": offset}], 10.)


def test_shift_gain_polarity_duplicate():
    x = np.random.default_rng(2027).normal(0, .1, 6000).astype(np.float32)
    y = np.r_[np.zeros(137), -.4 * x + .03, np.zeros(921)]
    assert module.max_correlation(x, y) > .9999
    assert module.max_correlation(x, np.random.default_rng(42).normal(0, .1, 12000)) < .1


def test_silence_is_not_near_duplicate():
    assert module.max_correlation(np.zeros(6000), np.zeros(10000)) == 0.
    assert module.fingerprints(np.zeros(10000))[0].shape == (0, 96)


def test_fingerprint_deterministic_gain_invariant():
    x = np.random.default_rng(2027).normal(0, .1, 10000).astype(np.float32)
    a, starts = module.fingerprints(x)
    b, _ = module.fingerprints(-.5 * x)
    assert starts == [0, 2000, 4000]
    np.testing.assert_allclose(a, b, atol=1e-5)
    np.testing.assert_array_equal(a, module.fingerprints(x)[0])


def test_pcm_hash_sensitive_to_content_not_array_layout():
    x = np.arange(100, dtype=np.float32)
    assert module.pcm_hash(x) == module.pcm_hash(x.copy())
    assert module.pcm_hash(x) != module.pcm_hash(x + 1)


def test_bad_archive_cannot_extract(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    archive.write_bytes(b"not official data")
    with pytest.raises(ValueError, match="size/MD5"):
        module.extract_verified(archive, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_missing_per_clip_license_is_not_eligible():
    assert not module.license_eligible(None)
    row = {"filename": "x.wav", "uploader": "author", "uploader_id": "id", "license": ""}
    assert not module.license_eligible(row)
    row["license"] = "Creative Commons Attribution license (reuse allowed)"
    assert module.license_eligible(row)
    row["uploader"] = ""
    assert not module.license_eligible(row)
