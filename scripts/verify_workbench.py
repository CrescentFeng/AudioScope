"""Exercise real inference, review persistence, exports and optional local CLAP/PG."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from audioscope.engineering import SEDDeployment
from audioscope.release_catalog import sessions
from audioscope.retrieval import ClapSearch, query_events
from audioscope.software_validation import write_json
from audioscope.workbench import AnalysisService, WorkbenchStore, playback, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--postgres", action="store_true")
    parser.add_argument("--clap", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    source = root / "data/l4-engineering/l4_long_demo.wav"
    model = root / "artifacts/l4-engineering/crnn_short_event_calibrated.pt"
    onnx = root / "artifacts/l4-engineering/crnn_short_event_fp32.onnx"
    url = sessions(root).kw["bind"].url if args.postgres else None
    with tempfile.TemporaryDirectory(prefix="audioscope-verification-") as folder:
        store = WorkbenchStore(Path(folder), url)
        job = AnalysisService(SEDDeployment(model, onnx), store).analyse(source)
        store.finish(job["id"], {**job["payload"], "verification_fixture": True})
        assert store.retained_audio(job["id"]) is None
        before = job["payload"]["events"]
        events = [
            {"label": e["label"], "onset_s": e["onset_s"], "offset_s": e["offset_s"]}
            for e in before
        ]
        # A verification fixture explicitly exercises review APIs; never represents human truth.
        revision = store.save_review(job["id"], events, 0, True)
        reopened = store.get(job["id"])
        assert reopened["review"]["source"] == "software_verification_fixture"
        assert reopened["payload"]["events"] == before
        files = store.export(job["id"])
        assert json.loads(Path(files[0]).read_text())["review"]["training_eligible"] is False
        result = query_events(before, "人声并伴随流水")
        assert result
        sample_rate, clip = playback(source, result[0])
        assert len(clip) > 0 and sample_rate == 16000
        report = {
            "status": "passed",
            "fixture": True,
            "not_human_annotation": True,
            "audio_sha256": sha256(source),
            "model": job["payload"]["model"],
            "job_id": job["id"],
            "database": "postgresql" if args.postgres else "sqlite",
            "events": len(before),
            "review_revision": revision,
            "overlap_hits": len(result),
            "default_raw_retention": False,
            "original_prediction_unchanged": True,
            "export_sha256": {Path(p).name: sha256(Path(p)) for p in files},
        }
        if args.clap:
            searcher = ClapSearch(root / "data/pretrained/clap-htsat-unfused-8fa0f1c")
            hits = searcher.search(source, "a person talking")
            repeat = searcher.search(source, "a person talking")
            assert all(np.isfinite(h["similarity"]) for h in hits)
            assert hits == repeat and len(hits) == 10
            report["clap"] = {
                "repeated_query_identical": True,
                "hits": hits,
                "retrieval_accuracy_evaluated": False,
            }
        destination = (
            root
            / "reports/generated/workbench"
            / ("postgres-verification.json" if args.postgres else "local-verification.json")
        )
        write_json(destination, report)
        print(json.dumps({"status": "passed", "report": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
