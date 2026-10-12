"""Exercise the real local container API and Gradio callbacks, not mocked models.

Only project demo audio is submitted. Automated reviews are explicitly fixtures.
Browser rendering/row-click verification remains a separate acceptance check.
"""

import argparse
import json
import subprocess
from pathlib import Path

import httpx
from gradio_client import Client, handle_file

from audioscope.model_registry import resolve_model
from audioscope.release_catalog import sessions
from audioscope.software_validation import write_json
from audioscope.workbench import WorkbenchStore, sha256


def run(root, api_url, ui_url):
    source = root / "data/l4-engineering/l4_long_demo.wav"
    expected_model, _ = resolve_model(root)
    store = WorkbenchStore(root / "runtime/workbench", sessions(root).kw["bind"].url)
    checks, job_ids = {}, []

    def fixture(job_id):
        job = store.get(job_id)
        store.finish(job_id, {**job["payload"], "verification_fixture": True})
        job_ids.append(job_id)
        return job

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise RuntimeError(f"Acceptance check failed: {name}")
        print(f"PASS {name}", flush=True)

    report = {"status": "running", "checks": checks, "fixture_job_ids": job_ids}
    destination = root / "reports/generated/workbench/container-verification.json"
    try:
        with httpx.Client(base_url=api_url, trust_env=False, timeout=300) as api:
            health = api.get("/health").raise_for_status().json()
            report["health"] = health
            check("migration_0002", health["revision"] == "0002_workbench")
            readiness = api.get("/v1/readiness").raise_for_status().json()
            report["catalog"] = readiness
            check("original_catalog_preserved", readiness["counts"]["audio_files"] == 3153)
            with source.open("rb") as audio:
                job = (
                    api.post("/v1/analysis", files={"audio": ("demo.wav", audio)})
                    .raise_for_status()
                    .json()
                )
            fixture(job["id"])
            check(
                "real_onnx_analysis",
                job["status"] == "completed" and bool(job["payload"]["events"]),
            )
            check(
                "registered_default_model",
                job["payload"]["model"]["checkpoint_sha256"] == sha256(expected_model),
            )
            check("default_no_raw_retention", store.retained_audio(job["id"]) is None)
            check(
                "history_persisted",
                api.get(f"/v1/analysis/{job['id']}").json()["payload"]
                == store.get(job["id"])["payload"],
            )
            check(
                "invalid_audio_rejected",
                api.post("/v1/analysis", files={"audio": ("bad.wav", b"invalid")}).status_code
                == 422,
            )
            review = {"events": [], "expected_revision": 0, "confirmed": False}
            route = f"/v1/analysis/{job['id']}/reviews"
            check("unconfirmed_review_rejected", api.post(route, json=review).status_code == 422)
            review["confirmed"] = True
            check("review_saved", api.post(route, json=review).json()["revision"] == 1)
            check("stale_review_rejected", api.post(route, json=review).status_code == 422)
            check(
                "review_not_training_or_human_truth",
                store.get(job["id"])["review"]["source"] == "software_verification_fixture"
                and not store.get(job["id"])["review"]["training_eligible"],
            )
            hits = (
                api.post(f"/v1/analysis/{job['id']}/search", json={"query": "人声并伴随流水"})
                .raise_for_status()
                .json()
            )
            check("api_overlap_search", len(hits) > 0)

        ui = Client(ui_url, verbose=False, httpx_kwargs={"trust_env": False})
        baseline = ui.predict(
            handle_file(str(source)), "overall", False, "v1_baseline", api_name="/analyse"
        )
        baseline_id = baseline[5]["value"]
        fixture(baseline_id)
        current = ui.predict(
            handle_file(str(source)), "overall", False, "current", api_name="/analyse"
        )
        current_id = current[5]["value"]
        original_job = fixture(current_id)
        check("ui_chinese_summary", "秒" in current[0])
        check(
            "ui_model_selector",
            current[4]["model"]["checkpoint_sha256"] != baseline[4]["model"]["checkpoint_sha256"],
        )
        comparison = ui.predict(baseline_id, api_name="/compare")
        check("same_audio_version_comparison", comparison[0]["right_id"] == baseline_id)
        reference = ui.predict(None, True, api_name="/reference_compare")
        check(
            "dataset_reference_comparison",
            reference[0]["source"] == "dataset_reference"
            and len(reference[0]["reference_events"]) > 0,
        )
        report["demo_reference_metrics"] = reference[0]["metrics"]
        results = ui.predict("人声并伴随流水", "已知类别（中文/英文）", True, api_name="/search")
        check("ui_overlap_search", bool(results[0]["data"]))
        semantic = ui.predict(
            "a person talking", "CLAP 开放描述（英文）", False, api_name="/search"
        )
        repeat = ui.predict("a person talking", "CLAP 开放描述（英文）", False, api_name="/search")
        check("container_local_clap", len(semantic[0]["data"]) == 10)
        check("clap_repeat_deterministic", semantic[0] == repeat[0])
        rows = current[3]
        saved = ui.predict(rows, True, api_name="/save")
        check(
            "ui_review_append", "v1" in saved[0] and saved[2] is False and "复核版本 1" in saved[3]
        )
        check(
            "prediction_immutable",
            store.get(current_id)["payload"]["events"] == original_job["payload"]["events"],
        )
        exported = ui.predict(api_name="/export")
        document = json.loads(Path(exported[0]).read_text())
        check(
            "json_csv_export",
            len(exported) == 2
            and document["id"] == current_id
            and document["review"]["source"] == "software_verification_fixture",
        )
        reopened = ui.predict(current_id, api_name="/reopen")
        check(
            "reopen_clears_stale_views",
            reopened[7] is False
            and reopened[8] is None
            and reopened[12] == {}
            and reopened[15] == {},
        )
        relink = ui.predict(handle_file(str(source)), api_name="/relink")
        check("hash_matched_relink", "已关联" in relink)
        report["image_id"] = subprocess.check_output(
            ["docker", "image", "inspect", "audioscope-python:local", "--format", "{{.Id}}"],
            text=True,
        ).strip()
        report["model_sha256"] = sha256(expected_model)
        report["status"] = "passed"
        report["browser_rendering_verified"] = False
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(destination, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8002")
    parser.add_argument("--ui", default="http://127.0.0.1:7862")
    args = parser.parse_args()
    run(Path.cwd(), args.api, args.ui)
