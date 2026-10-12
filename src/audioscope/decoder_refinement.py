"""Validation-only boundary refinement of the frozen v1.0 acoustic model."""

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch

from .audio import read_wav_mono
from .engineering import SEDDeployment
from .features import extract_logmel
from .sed import _evaluate_probabilities, _select_class_thresholds
from .sed_data import read_soundscape_manifest
from .sed_enhanced import _select_short_event_threshold
from .software_validation import CONDITIONS, MODEL, ONNX, transform, write_json
from .workbench import sha256


def run(root):
    manifest = root / "data/desed-real-strong-only/manifest.jsonl"
    records = [r for r in read_soundscape_manifest(manifest) if r.split == "validation"]
    deployment = SEDDeployment(root / MODEL, root / ONNX)
    artifact = deployment.artifact
    metric_settings = {
        "event": {"onset_collar_s": 0.2, "offset_collar_s": 0.2, "offset_ratio": 0.2},
        "segment_resolution_s": 1.0,
    }
    report_root = root / "reports/generated/model-improvement"
    protocol = {
        "source_sha256": sha256(root / MODEL),
        "manifest_sha256": sha256(manifest),
        "split": "validation",
        "clips": len(records),
        "test_used": False,
        "thresholds": [0.8, 0.9],
        "smooth_frames": [3, 5],
        "merge_gap_s": [0.1, 0.2, 0.3],
        "minimum_duration_s": [0.06, 0.12],
        "conditions": CONDITIONS,
        "objective": "mean condition micro/macro event F1",
        "clean_guard": "clean micro F1 >= original - 0.01",
        "tie_break": "prefer original; otherwise deterministic grid order",
    }
    write_json(report_root / "decoder_protocol.json", protocol)
    probabilities = {}
    for condition in CONDITIONS:
        probabilities[condition] = np.stack(
            [
                deployment.predict_feature(
                    extract_logmel(
                        transform(
                            read_wav_mono(manifest.parent / r.path, 16000)[0],
                            16000,
                            condition,
                            8127 + i,
                        ),
                        16000,
                        deployment.feature_settings,
                    )
                )
                for i, r in enumerate(records)
            ]
        )

    def score(threshold, settings):
        metrics = {
            condition: _evaluate_probabilities(
                records, values, deployment.class_names, threshold, 0.02, settings, metric_settings
            )[0]
            for condition, values in probabilities.items()
        }
        objective = float(
            np.mean(
                [
                    (m["event_based"]["micro"]["f1"] + m["event_based"]["macro_f1"]) / 2
                    for m in metrics.values()
                ]
            )
        )
        return {
            "threshold": threshold,
            "postprocess": settings,
            "metrics": metrics,
            "objective": objective,
        }

    original = score(artifact["threshold_profiles"]["overall"], artifact["postprocess_settings"])
    rows = [original]
    for threshold, smooth, gap, minimum in itertools.product(
        protocol["thresholds"],
        protocol["smooth_frames"],
        protocol["merge_gap_s"],
        protocol["minimum_duration_s"],
    ):
        settings = {
            **artifact["postprocess_settings"],
            "smooth_frames": smooth,
            "merge_gap_s": gap,
            "minimum_duration_s": minimum,
        }
        row = score(threshold, settings)
        rows.append(row)
        print(f"Decoder {threshold}/{smooth}/{gap}/{minimum}: {row['objective']:.4f}", flush=True)
    clean_floor = original["metrics"]["clean"]["event_based"]["micro"]["f1"] - 0.01
    eligible = [
        r for r in rows if r["metrics"]["clean"]["event_based"]["micro"]["f1"] >= clean_floor
    ]
    winner = max(eligible, key=lambda r: r["objective"])
    grid = artifact["postprocess_settings"]["threshold_grid"]
    class_thresholds, _ = _select_class_thresholds(
        records,
        probabilities["clean"],
        deployment.class_names,
        grid,
        0.02,
        winner["postprocess"],
        metric_settings,
    )
    short, _ = _select_short_event_threshold(
        records,
        probabilities["clean"],
        deployment.class_names,
        grid,
        0.02,
        winner["postprocess"],
        metric_settings,
        1.0,
    )
    checkpoint = {
        **artifact,
        "postprocess_settings": winner["postprocess"],
        "threshold": winner["threshold"],
        "threshold_profiles": {
            "overall": winner["threshold"],
            "class_balanced": class_thresholds,
            "short_event": short,
        },
        "refinement": {"source_sha256": protocol["source_sha256"], "validation_only": True},
    }
    path = root / "artifacts/model-improvement/crnn_postprocess_candidate.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    write_json(
        report_root / "decoder_selection.json",
        {
            "status": "completed",
            "test_used": False,
            "protocol": protocol,
            "original": original,
            "selected": winner,
            "all_candidates": rows,
            "candidate": str(path),
            "candidate_sha256": sha256(path),
            "onnx_sha256": sha256(root / ONNX),
            "default_changed": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    run(parser.parse_args().root.resolve())
