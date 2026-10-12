from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import chirp, correlate, correlation_lags

from .audio import read_wav_mono


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _balanced_test_selection(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    candidates = [record for record in records if record["split"] == "test"]
    candidates.sort(key=lambda row: row["scene_id"])
    selected: list[dict[str, Any]] = []
    covered: Counter[str] = Counter()
    while candidates and len(selected) < count:
        best = max(
            candidates,
            key=lambda row: (
                sum(1.0 / (1.0 + covered[event["label"]]) for event in row["events"]),
                len({event["label"] for event in row["events"]}),
                row["scene_id"],
            ),
        )
        candidates.remove(best)
        selected.append(best)
        covered.update({event["label"] for event in best["events"]})
    return selected


def _sync_marker(sample_rate: int, duration_s: float = 0.75) -> np.ndarray:
    times = np.arange(round(sample_rate * duration_s), dtype=np.float32) / sample_rate
    envelope = np.sin(np.pi * np.minimum(times / 0.03, 1.0))
    tail = np.sin(np.pi * np.minimum((duration_s - times) / 0.03, 1.0))
    return (0.35 * chirp(times, f0=800, f1=3200, t1=duration_s) * envelope * tail).astype(
        np.float32
    )


def _normalize_clip(samples: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    if rms <= 1e-8:
        return samples.astype(np.float32)
    target_rms = 10 ** (target_dbfs / 20.0)
    scaled = samples * min(target_rms / rms, 0.98 / max(float(np.max(np.abs(samples))), 1e-8))
    return scaled.astype(np.float32)


def prepare_field_kit(root: Path, count: int = 20) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "data/desed-real-strong-only/manifest.jsonl"
    dataset_root = manifest_path.parent
    output = root / "data/l4-field-kit"
    output.mkdir(parents=True, exist_ok=True)
    selected = _balanced_test_selection(_load_manifest(manifest_path), count)
    sample_rate = 16000
    marker = _sync_marker(sample_rate)
    one_second = np.zeros(sample_rate, dtype=np.float32)
    quarter_second = np.zeros(sample_rate // 4, dtype=np.float32)
    pieces = [one_second]
    cue_rows = []
    cursor = len(one_second)
    for index, record in enumerate(selected, start=1):
        pieces.extend([marker, quarter_second])
        cursor += len(marker) + len(quarter_second)
        samples, _ = read_wav_mono(dataset_root / record["path"], sample_rate)
        samples = _normalize_clip(samples)
        onset_s = cursor / sample_rate
        pieces.extend([samples, one_second])
        cursor += len(samples) + len(one_second)
        cue_rows.append(
            {
                "clip_index": index,
                "scene_id": record["scene_id"],
                "reference_onset_s": round(onset_s, 6),
                "duration_s": round(len(samples) / sample_rate, 6),
                "labels": sorted({event["label"] for event in record["events"]}),
                "events": record["events"],
                "source_path": record["path"],
            }
        )
    playback = np.concatenate(pieces)
    playback_path = output / "reference_playback.wav"
    wavfile.write(playback_path, sample_rate, np.round(playback * 32767).astype(np.int16))
    cue_path = output / "reference_manifest.jsonl"
    cue_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cue_rows),
        encoding="utf-8",
    )
    matrix_path = output / "recording_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "session_id",
            "device_name",
            "room_name",
            "distance_m",
            "playback_level_db",
            "recording_path",
            "status",
            "operator_notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for device in range(1, 4):
            for room in range(1, 4):
                session_id = f"device-{device}_room-{room}"
                writer.writerow(
                    {
                        "session_id": session_id,
                        "device_name": f"待填写设备{device}",
                        "room_name": f"待填写房间{room}",
                        "distance_m": "",
                        "playback_level_db": "",
                        "recording_path": f"recordings/{session_id}.wav",
                        "status": "planned",
                        "operator_notes": "",
                    }
                )
    class_counts = Counter(label for row in cue_rows for label in row["labels"])
    summary = {
        "status": "prepared",
        "sample_rate": sample_rate,
        "clips": len(cue_rows),
        "sessions": 9,
        "derived_recordings_after_capture": 9 * len(cue_rows),
        "playback_duration_s": len(playback) / sample_rate,
        "playback_sha256": _sha256(playback_path),
        "class_clip_counts": dict(sorted(class_counts.items())),
        "paths": {
            "playback": str(playback_path),
            "reference_manifest": str(cue_path),
            "recording_matrix": str(matrix_path),
        },
    }
    (output / "field_kit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def estimate_sync_offset(recording: np.ndarray, reference: np.ndarray, sample_rate: int) -> float:
    probe = reference[: min(len(reference), sample_rate * 3)]
    search = recording[: min(len(recording), sample_rate * 15)]
    if len(search) < len(probe):
        raise ValueError("recording is shorter than the synchronization probe")
    values = correlate(search, probe, mode="valid", method="fft")
    lags = correlation_lags(len(search), len(probe), mode="valid")
    return float(lags[int(np.argmax(values))] / sample_rate)


def _recording_quality(samples: np.ndarray, sample_rate: int, expected_s: float) -> dict[str, Any]:
    duration_s = len(samples) / sample_rate
    absolute = np.abs(samples)
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    clipping_ratio = float(np.mean(absolute >= 0.999))
    peak = float(absolute.max(initial=0.0))
    checks = {
        "duration": duration_s >= expected_s * 0.98,
        "not_clipped": clipping_ratio <= 0.001,
        "not_silent": rms >= 10 ** (-55 / 20),
    }
    return {
        "duration_s": duration_s,
        "expected_duration_s": expected_s,
        "rms_dbfs": 20 * math.log10(max(rms, 1e-9)),
        "peak_dbfs": 20 * math.log10(max(peak, 1e-9)),
        "clipping_ratio": clipping_ratio,
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
    }


def run_field_qc(root: Path, model_path: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    kit = root / "data/l4-field-kit"
    reference, sample_rate = read_wav_mono(kit / "reference_playback.wav", 16000)
    cues = _load_manifest(kit / "reference_manifest.jsonl")
    sessions = list(csv.DictReader((kit / "recording_matrix.csv").open(encoding="utf-8")))
    deployment = None
    if model_path:
        from .engineering import SEDDeployment

        deployment = SEDDeployment(model_path.resolve())
    rows = []
    totals = Counter()
    for session in sessions:
        recording_path = kit / session["recording_path"]
        if not recording_path.exists():
            rows.append({"session_id": session["session_id"], "status": "missing"})
            continue
        recording, _ = read_wav_mono(recording_path, sample_rate)
        offset_s = estimate_sync_offset(recording, reference, sample_rate)
        quality = _recording_quality(recording, sample_rate, len(reference) / sample_rate)
        session_result: dict[str, Any] = {
            "session_id": session["session_id"],
            "status": quality["status"],
            "recording_sha256": _sha256(recording_path),
            "sync_offset_s": offset_s,
            "quality": quality,
        }
        if deployment:
            tp = fp = fn = 0
            per_clip = []
            for cue in cues:
                left = round((offset_s + float(cue["reference_onset_s"])) * sample_rate)
                right = left + round(float(cue["duration_s"]) * sample_rate)
                if left < 0 or right > len(recording):
                    continue
                prediction = deployment.predict_samples(recording[left:right])
                expected = set(cue["labels"])
                observed = {event["label"] for event in prediction["events"]}
                tp += len(expected & observed)
                fp += len(observed - expected)
                fn += len(expected - observed)
                per_clip.append(
                    {
                        "scene_id": cue["scene_id"],
                        "expected": sorted(expected),
                        "observed": sorted(observed),
                    }
                )
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            session_result["recognition"] = {
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / max(precision + recall, 1e-9),
                "clips": per_clip,
            }
            totals.update(tp=tp, fp=fp, fn=fn)
        rows.append(session_result)
    captured = sum(row["status"] != "missing" for row in rows)
    passed = sum(row["status"] == "passed" for row in rows)
    result: dict[str, Any] = {
        "status": "complete"
        if captured == len(rows) and passed == captured
        else "awaiting_field_capture",
        "planned_sessions": len(rows),
        "captured_sessions": captured,
        "quality_passed_sessions": passed,
        "sessions": rows,
    }
    if deployment and totals:
        precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
        recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
        result["aggregate_recognition"] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        }
    report_dir = root / "reports/generated/field"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "field_qc.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and evaluate AudioScope field sessions")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--root", type=Path, default=Path("."))
    prepare.add_argument("--clips", type=int, default=20)
    qc = subparsers.add_parser("qc")
    qc.add_argument("--root", type=Path, default=Path("."))
    qc.add_argument("--model", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_field_kit(args.root, args.clips)
    else:
        result = run_field_qc(args.root, args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
