"""Evaluate frozen candidates, including negative results; never adjust on test results."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .audio import read_wav_mono
from .engineering import SEDDeployment
from .event_review import error_triage
from .sed_data import TimedEvent, read_soundscape_manifest
from .sed_metrics import duration_binned_event_metrics, evaluate_sed
from .software_validation import CONDITIONS, MODEL, ONNX, transform, write_json
from .workbench import sha256


def run(root):
    torch.set_num_threads(4)
    output = root / "reports/generated/model-improvement"
    development = json.loads((output / "development_summary.json").read_text())
    decoder = json.loads((output / "decoder_selection.json").read_text())
    assert development["status"] == decoder["status"] == "completed"
    paths = {"v1_baseline": root / MODEL, "decoder_candidate": Path(decoder["candidate"])}
    assert sha256(paths["decoder_candidate"]) == decoder["candidate_sha256"]
    for variant in ("paired_baseline", "balanced_domain_augmented"):
        for seed in (2027, 2028, 2029):
            key = f"{variant}-{seed}"
            report = json.loads((output / f"{key}.json").read_text())
            paths[key] = Path(report["checkpoint"])
            assert report["protocol_signature"] == development["protocol_signature"]
            assert sha256(paths[key]) == report["sha256"]
    manifest = root / "data/desed-real-strong-only/manifest.jsonl"
    records = read_soundscape_manifest(manifest)
    split_sources = {
        split: {e.source_recording_id for r in records if r.split == split for e in r.events}
        for split in ("train", "validation", "test")
    }
    assert not split_sources["test"] & (split_sources["train"] | split_sources["validation"])
    records = [r for r in records if r.split == "test"]
    protocol = {
        "clips": len(records),
        "manifest_sha256": sha256(manifest),
        "development_sha256": sha256(output / "development_summary.json"),
        "decoder_selection_sha256": sha256(output / "decoder_selection.json"),
        "models": {key: sha256(path) for key, path in paths.items()},
        "conditions": CONDITIONS,
        "seed": 2027,
        "threshold_selection": "validation_only",
        "default_promotion_gate": "decoder clean micro >= v1, clean macro >= v1-0.01, "
        "mean perturbed micro >= v1, short-event clean F1 >= v1-0.02",
        "caveat": "Previously published test split, not a fresh blind benchmark.",
    }
    protocol_file = output / "heldout_protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError("Frozen held-out protocol changed; cannot silently reselect on test")
    write_json(protocol_file, protocol)
    deployments = {
        key: SEDDeployment(
            path, root / ONNX if key in {"v1_baseline", "decoder_candidate"} else None
        )
        for key, path in paths.items()
    }
    event_settings = {"onset_collar_s": 0.2, "offset_collar_s": 0.2, "offset_ratio": 0.2}
    rows = []
    for condition in CONDITIONS:
        for key, deployment in deployments.items():
            row_path = output / "heldout" / f"{key}-{condition}.json"
            if row_path.exists():
                row = json.loads(row_path.read_text())
                if row.get("protocol_sha256") == sha256(protocol_file):
                    rows.append(row)
                    continue
            estimates, triage_rows, elapsed = {}, [], []
            for i, record in enumerate(records):
                samples, _ = read_wav_mono(manifest.parent / record.path, 16000)
                samples = transform(samples, 16000, condition, 2027 + i)
                start = time.perf_counter()
                result = deployment.predict_samples(samples, include_unknown=False)
                elapsed.append(time.perf_counter() - start)
                predictions = result["events"]
                estimates[record.scene_id] = [
                    TimedEvent(e["label"], e["onset_s"], e["offset_s"], "prediction", "prediction")
                    for e in predictions
                ]
                reference = [
                    {"label": e.label, "onset_s": e.onset_s, "offset_s": e.offset_s}
                    for e in record.events
                ]
                triage_rows.append(
                    {"scene_id": record.scene_id, **error_triage(reference, predictions)}
                )
            metrics = evaluate_sed(records, estimates, deployment.class_names, event_settings, 1.0)
            durations = duration_binned_event_metrics(
                records, estimates, deployment.class_names, event_settings, 1.0
            )
            row = {
                "model": key,
                "condition": condition,
                "protocol_sha256": sha256(protocol_file),
                "metrics": metrics,
                "duration_metrics": durations,
                "inference_rtf": sum(elapsed) / sum(r.duration_s for r in records),
                "clip_compute_p95_ms": float(np.percentile(elapsed, 95) * 1000),
                "fn_categories": dict(
                    Counter(e["category"] for r in triage_rows for e in r["missed_reference"])
                ),
                "fp_categories": dict(
                    Counter(e["category"] for r in triage_rows for e in r["extra_prediction"])
                ),
            }
            write_json(row_path, row)
            if condition == "clean" and key in {"v1_baseline", "decoder_candidate"}:
                write_json(output / f"{key}_error_cases.json", triage_rows)
            rows.append(row)
            print(
                f"{key}/{condition}: event-F1={metrics['event_based']['micro']['f1']:.4f}",
                flush=True,
            )
    baseline = next(r for r in rows if r["model"] == "v1_baseline" and r["condition"] == "clean")
    candidate = next(
        r for r in rows if r["model"] == "decoder_candidate" and r["condition"] == "clean"
    )
    shifted = {
        key: float(
            np.mean(
                [
                    r["metrics"]["event_based"]["micro"]["f1"]
                    for r in rows
                    if r["model"] == key and r["condition"] != "clean"
                ]
            )
        )
        for key in ("v1_baseline", "decoder_candidate")
    }
    checks = {
        "clean_micro": candidate["metrics"]["event_based"]["micro"]["f1"]
        >= baseline["metrics"]["event_based"]["micro"]["f1"],
        "clean_macro": candidate["metrics"]["event_based"]["macro_f1"]
        >= baseline["metrics"]["event_based"]["macro_f1"] - 0.01,
        "mean_shift_micro": shifted["decoder_candidate"] >= shifted["v1_baseline"],
        "short_event_guard": candidate["duration_metrics"]["short"]["micro"]["f1"]
        >= baseline["duration_metrics"]["short"]["micro"]["f1"] - 0.02,
    }
    write_json(
        output / "heldout_summary.json",
        {
            "status": "completed",
            "protocol": protocol,
            "results": rows,
            "promotion_checks": checks,
            "quality_gate_passed": all(checks.values()),
            "default_changed": False,
            "mean_shift_micro": shifted,
            "notice": "Quality gate is not deployment/streaming regression acceptance.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    run(parser.parse_args().root.resolve())
