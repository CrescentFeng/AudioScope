from __future__ import annotations

import json
import socket
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from sqlalchemy import select

from audioscope.launch import available_port, bypass_loopback_proxy
from audioscope.retrieval import query_events
from audioscope.workbench import (
    AnalysisService,
    WorkbenchStore,
    playback,
    reviews,
    validate_audio,
    validate_events,
)


def event(label="speech", start=0.1, end=0.8):
    return {"label": label, "onset_s": start, "offset_s": end, "confidence": 0.9}


class FakeDeployment:
    sample_rate = 16000
    provenance = {"checkpoint_sha256": "fake"}

    def predict_samples(self, samples, threshold_profile):
        return {
            "events": [event()],
            "unknown_events": [],
            "backend": "test",
            "stats": {"audio_seconds": len(samples) / self.sample_rate},
        }


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "input.wav"
    wavfile.write(path, 16000, np.zeros(16000, dtype=np.int16))
    return path


def test_analysis_default_does_not_retain_audio_and_persists_history(tmp_path, audio):
    store = WorkbenchStore(tmp_path / "store")
    job = AnalysisService(FakeDeployment(), store).analyse(audio)
    assert store.retained_audio(job["id"]) is None
    reopened = WorkbenchStore(tmp_path / "store").get(job["id"])
    assert reopened["payload"]["events"] == [event()]
    assert reopened["status"] == "completed"
    assert str(audio) not in json.dumps(reopened)
    assert reopened["payload"]["training_eligible"] is False


def test_explicit_retention_and_exports(tmp_path, audio):
    store = WorkbenchStore(tmp_path / "store")
    job = AnalysisService(FakeDeployment(), store).analyse(audio, retain_audio=True)
    assert store.retained_audio(job["id"]).read_bytes() == audio.read_bytes()
    store.save_review(job["id"], [event("dog")], 0, True)
    files = store.export(job["id"])
    output = json.loads(Path(files[0]).read_text())
    assert output["payload"]["events"][0]["label"] == "speech"
    assert output["review"]["events"][0]["label"] == "dog"
    assert "human_review" in Path(files[1]).read_text(encoding="utf-8-sig")
    assert store.cleanup_exports(now=10**12) == 1
    assert store.retained_audio(job["id"]).is_file()


def test_review_requires_confirmation_validates_bounds_and_optimistic_lock(tmp_path, audio):
    store = WorkbenchStore(tmp_path / "store")
    job = AnalysisService(FakeDeployment(), store).analyse(audio)
    with pytest.raises(ValueError, match="确认"):
        store.save_review(job["id"], [], 0, False)
    assert store.save_review(job["id"], [event("cat")], 0, True) == 1
    with pytest.raises(ValueError, match="其他会话"):
        store.save_review(job["id"], [], 0, True)
    assert store.save_review(job["id"], [], 1, True) == 2
    with store.engine.connect() as conn:
        assert len(conn.execute(select(reviews)).all()) == 2
    assert store.get(job["id"])["payload"]["events"] == [event()]


@pytest.mark.parametrize(
    "item",
    [
        event(start=-1),
        event(end=2),
        event(end=float("nan")),
        event(start=0.8, end=0.8),
        event("invented"),
    ],
)
def test_invalid_review(item):
    with pytest.raises(ValueError):
        validate_events([item], 1)


def test_playback_context_is_clamped(audio):
    sr, samples = playback(audio, event(start=0, end=0.8))
    assert sr == 16000
    assert len(samples) == 16000


def test_float_wav_and_nonfinite_samples(tmp_path):
    path = tmp_path / "float.wav"
    wavfile.write(path, 16000, np.zeros(16000, dtype=np.float32))
    assert validate_audio(path) == 1
    wavfile.write(path, 16000, np.array([np.inf, np.nan], dtype=np.float32))
    with pytest.raises(ValueError, match="NaN/Inf"):
        validate_audio(path)


def test_limits_and_failed_job(tmp_path, audio):
    with pytest.raises(ValueError):
        validate_audio(audio, max_seconds=0.5)
    with pytest.raises(ValueError):
        validate_audio(audio, max_bytes=10)

    class Failing(FakeDeployment):
        def predict_samples(self, *args, **kwargs):
            raise RuntimeError("private debug detail")

    store = WorkbenchStore(tmp_path / "store")
    with pytest.raises(RuntimeError):
        AnalysisService(Failing(), store).analyse(audio)
    job = store.history()[0]
    assert job["status"] == "failed"
    assert "private debug" not in json.dumps(job)


def test_known_query_overlap_and_empty_results():
    events = [event("speech", 0, 2), event("running_water", 1, 3), event("dog", 0, 1)]
    hits = query_events(events, "说话并伴随流水")
    assert len(hits) == 1 and hits[0]["onset_s"] == 1 and hits[0]["offset_s"] == 2
    assert len(query_events(events, "人声 水声")) == 2
    assert query_events(events, "猫叫") == []
    with pytest.raises(ValueError):
        query_events(events, "没有人声")
    with pytest.raises(ValueError):
        query_events(events, "金属碰撞")


def test_launcher_preserves_existing_no_proxy(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "internal.example")
    bypass_loopback_proxy()
    import os

    assert "internal.example" in os.environ["NO_PROXY"]
    assert "127.0.0.1" in os.environ["NO_PROXY"]


def test_launcher_never_reuses_occupied_port():
    with socket.socket() as occupied:
        try:
            occupied.bind(("127.0.0.1", 0))
        except PermissionError:
            pytest.skip("sandbox does not permit loopback bind")
        port = occupied.getsockname()[1]
        with pytest.raises(RuntimeError):
            available_port(port, 1)
