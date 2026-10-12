import json
from pathlib import Path

from audioscope.release_audit import audit


def _write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_preflight_excludes_manual_checks(tmp_path):
    required = [
        "migrations/versions/0001_audioscope_catalog.py",
        "compose.yaml",
        "src/audioscope/api.py",
        "docs/PROJECT_SUMMARY_BILINGUAL.md",
        "docs/DEMO_VIDEO_SCRIPT_BILINGUAL.md",
        "docs/DATABASE_AND_DOCKER.md",
    ]
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ready", encoding="utf-8")
    _write(tmp_path / "reports/generated/database/import_report.json", {"status": "passed"})
    _write(tmp_path / "reports/generated/docker/validation.json", {"status": "passed"})
    _write(tmp_path / "data/l4-field-kit/field_kit_summary.json", {"status": "prepared"})
    _write(tmp_path / "reports/generated/field/mic_simulation.json", {"status": "passed"})
    result = audit(tmp_path, "preflight")
    assert result["status"] == "passed"
    assert result["non_field_complete"] is True
    assert result["final_release_complete"] is False
    assert result["software_release_complete"] is False
    assert result["physical_validation_performed"] is False


def test_final_alias_keeps_field_requirements(tmp_path):
    for mode in ("final", "field-final", "software-final"):
        result = audit(tmp_path, mode)
        assert result["status"] == "blocked"
        assert not result["field_release_complete"]
    _write(
        tmp_path / "reports/generated/field/mic_live_acceptance.json",
        {"status": "passed", "mode": "file_simulation", "hardware_validated": False},
    )
    _write(tmp_path / "reports/generated/field/field_qc.json", {"status": "complete"})
    assert not audit(tmp_path, "field-final")["physical_validation_performed"]
