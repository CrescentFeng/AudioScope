import io

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scipy.io import wavfile

from audioscope import workbench_api
from audioscope.workbench import AnalysisService, WorkbenchStore


class Deployment:
    sample_rate = 16000
    provenance = {"checkpoint_sha256": "test"}

    def predict_samples(self, samples, threshold_profile):
        return {
            "events": [{"label": "speech", "onset_s": 0.0, "offset_s": 0.5}],
            "unknown_events": [],
            "stats": {},
            "backend": "test",
        }


def test_api_analysis_history_review_and_search(tmp_path, monkeypatch):
    monkeypatch.setattr(
        workbench_api, "_service", AnalysisService(Deployment(), WorkbenchStore(tmp_path))
    )
    app = FastAPI()
    app.include_router(workbench_api.router)
    with TestClient(app) as client:
        audio = io.BytesIO()
        wavfile.write(audio, 16000, np.zeros(16000, np.int16))
        response = client.post("/v1/analysis", files={"audio": ("test.wav", audio.getvalue())})
        assert response.status_code == 200
        job_id = response.json()["id"]
        assert client.get(f"/v1/analysis/{job_id}").json()["revision"] == 0
        assert len(client.get("/v1/analysis").json()) == 1
        response = client.post(f"/v1/analysis/{job_id}/search", json={"query": "人声"})
        assert response.status_code == 200 and len(response.json()) == 1
        assert (
            client.post(
                f"/v1/analysis/{job_id}/search", json={"query": "talking", "semantic": True}
            ).status_code
            == 422
        )
        assert (
            client.post(
                f"/v1/analysis/{job_id}/reviews", json={"events": [], "expected_revision": 0}
            ).status_code
            == 422
        )
        assert (
            client.post(
                f"/v1/analysis/{job_id}/reviews",
                json={"events": [], "expected_revision": 0, "confirmed": True},
            ).status_code
            == 200
        )
        assert client.get(f"/v1/analysis/{job_id}").json()["review"]["events"] == []
        assert client.get("/v1/analysis/not-a-job").status_code == 404
        assert (
            client.post("/v1/analysis", files={"audio": ("bad.wav", b"not audio")}).status_code
            == 422
        )
