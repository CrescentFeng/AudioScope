import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("external_collector", ROOT / "scripts/collect_external_pilot.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def test_selection_is_ten_distinct_classes_and_no_model_selection():
    spec = json.loads((collector.OUT / "selection.json").read_text())
    assert len(spec["items"]) == 10
    assert {r["label"] for r in spec["items"]} == collector.LABELS
    assert len({r["id"] for r in spec["items"]}) == 10
    speech = next(r for r in spec["items"] if r["id"].startswith("commons"))
    assert speech["label"] == "speech"


def test_license_and_preview_must_be_published():
    item = {"id": "fs-123", "platform": "freesound", "license_url": "https://creativecommons.org/publicdomain/zero/1.0/"}
    license_html = '<a href="http://creativecommons.org/publicdomain/zero/1.0/">CC0</a>'
    preview = '<div data-static-file-url="https://cdn.freesound.org/previews/0/123_45-hq.mp3">'
    assert collector.resolve_media(item, license_html + preview)[0].endswith("123_45-hq.mp3")
    for bad in [preview, license_html, license_html + preview.replace("123_", "999_")]:
        with pytest.raises(AssertionError):
            collector.resolve_media(item, bad)


def test_downloaded_drafts_cannot_be_reference_truth():
    from audioscope.event_review import compare_reference
    drafts = list((collector.OUT / "annotation-drafts").glob("*.json"))
    if not drafts:
        pytest.skip("Media not collected; network is never required in CI")
    for p in drafts:
        d = json.loads(p.read_text())
        assert d["complete"] is False and d["events"] is None
        assert d["human_listened"] is False
        job = {"payload": {"audio_sha256": d["audio_sha256"]}}
        with pytest.raises(ValueError, match="参考来源"):
            compare_reference(job, d)
