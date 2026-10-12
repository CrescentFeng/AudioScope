import json

import pytest

from audioscope.model_registry import REGISTRY, resolve_model
from audioscope.workbench import sha256
from audioscope.workbench_ui import history_database_url, selected_event


def test_sorted_table_selection_uses_values_not_position():
    events = [
        {"label": "speech", "onset_s": 0, "offset_s": 1},
        {"label": "dog", "onset_s": 2, "offset_s": 3},
    ]
    assert selected_event(events, ["dog", 2, 3, 0.8]) == events[1]
    with pytest.raises(ValueError):
        selected_event(events, ["cat", 2, 3])
    with pytest.raises(ValueError):
        selected_event(events, None)


def test_explicit_offline_overrides_database_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIOSCOPE_DATABASE_URL", "invalid-database")
    monkeypatch.setenv("AUDIOSCOPE_WORKBENCH_DB_URL", "another-invalid-database")
    monkeypatch.setenv("AUDIOSCOPE_LOCAL_HISTORY", "1")
    assert history_database_url(tmp_path) is None
    monkeypatch.delenv("AUDIOSCOPE_LOCAL_HISTORY")
    assert history_database_url(tmp_path) == "another-invalid-database"


def test_model_registry_rejects_tamper_and_path_escape(tmp_path):
    (tmp_path / REGISTRY).parent.mkdir(parents=True)
    weights = tmp_path / "artifacts/model.pt"
    weights.write_bytes(b"fixture")
    entry = {
        "checkpoint": "artifacts/model.pt",
        "onnx": "artifacts/model.pt",
        "checkpoint_sha256": sha256(weights),
    }
    registry = {"default": "test", "models": {"test": entry}}
    (tmp_path / REGISTRY).write_text(json.dumps(registry))
    assert resolve_model(tmp_path) == (weights, weights)
    with pytest.raises(ValueError):
        resolve_model(tmp_path, "unknown")
    weights.write_bytes(b"modified")
    with pytest.raises(ValueError, match="哈希"):
        resolve_model(tmp_path)
    entry["checkpoint"] = "../outside.pt"
    (tmp_path / REGISTRY).write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="目录"):
        resolve_model(tmp_path)
