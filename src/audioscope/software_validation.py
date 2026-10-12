"""Reproducible software domain shifts and accelerated streaming endurance checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import psutil
from scipy.signal import butter, fftconvolve, sosfilt

from .audio import read_wav_mono
from .engineering import SEDDeployment, StreamingSEDProcessor
from .release_audit import _sha256
from .sed_data import TimedEvent, read_soundscape_manifest
from .sed_metrics import evaluate_sed

CONDITIONS = ["clean", "reverb_025", "reverb_080", "telephone", "laptop", "pink_10db", "combined"]
OUT = Path("reports/generated/software-release")
MODEL = Path("artifacts/l4-engineering/crnn_short_event_calibrated.pt")
ONNX = Path("artifacts/l4-engineering/crnn_short_event_fp32.onnx")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def transform(samples, sample_rate, condition, seed=2027):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    rng = np.random.default_rng(seed)
    result = np.asarray(samples, dtype=np.float64).copy()
    if condition == "clean":
        return result.astype(np.float32)
    if condition.startswith("reverb") or condition == "combined":
        rt60 = 0.25 if condition == "reverb_025" else 0.8
        t = np.arange(round(rt60 * sample_rate)) / sample_rate
        impulse = rng.normal(size=len(t)) * np.exp(-np.log(1000) * t / rt60)
        impulse[0] = 0
        impulse *= 0.4 / max(np.linalg.norm(impulse), 1e-12)
        impulse[0] = 1
        result = fftconvolve(result, impulse)[: len(result)]
    if condition in {"telephone", "laptop", "combined"}:
        band = [300, 3400] if condition == "telephone" else [180, 6000]
        result = sosfilt(butter(4, band, btype="bandpass", fs=sample_rate, output="sos"), result)
        if condition != "telephone":
            result = np.tanh(1.2 * result) / 1.2
    if condition in {"pink_10db", "combined"}:
        freq = np.fft.rfftfreq(len(result), 1 / sample_rate)
        noise = np.fft.irfft(
            np.fft.rfft(rng.normal(size=len(result))) / np.sqrt(np.maximum(freq, 20)),
            n=len(result),
        )
        noise -= noise.mean()
        noise *= np.sqrt(np.mean(result**2) / max(np.mean(noise**2), 1e-12) / 10)
        result += noise
    # Explicit peak limiter preserves length; scaling is part of the frozen protocol.
    result *= min(1.0, 0.98 / max(float(np.max(np.abs(result))), 1e-12))
    return result.astype(np.float32)


def evaluate(root):
    manifest = root / "data/desed-real-strong-only/manifest.jsonl"
    records = [r for r in read_soundscape_manifest(manifest) if r.split == "test"]
    if len(records) != 221:
        raise ValueError(f"Expected 221 test recordings, found {len(records)}")
    deployment = SEDDeployment(root / MODEL, root / ONNX)
    rows = []
    event_settings = {"onset_collar_s": 0.2, "offset_collar_s": 0.2, "offset_ratio": 0.2}
    for condition in CONDITIONS:
        estimates = {}
        hashes, repeat_hashes = [], []
        model_seconds = 0.0
        started = time.perf_counter()
        for index, record in enumerate(records):
            samples, _ = read_wav_mono(manifest.parent / record.path, deployment.sample_rate)
            transformed = transform(samples, deployment.sample_rate, condition, 2027 + index)
            repeated = transform(samples, deployment.sample_rate, condition, 2027 + index)
            if not np.array_equal(transformed, repeated) or not np.isfinite(transformed).all():
                raise ValueError("Non-reproducible or non-finite transformation")
            prediction = deployment.predict_samples(transformed, include_unknown=False)
            verification = deployment.predict_samples(repeated, include_unknown=False)
            hashes.append(fingerprint(prediction["events"]))
            repeat_hashes.append(fingerprint(verification["events"]))
            model_seconds += prediction["stats"]["model_seconds"]
            estimates[record.scene_id] = [
                TimedEvent(
                    label=e["label"],
                    onset_s=e["onset_s"],
                    offset_s=e["offset_s"],
                    source_asset_id="prediction",
                    source_recording_id="prediction",
                )
                for e in prediction["events"]
            ]
        metrics = evaluate_sed(records, estimates, deployment.class_names, event_settings, 1.0)
        elapsed = time.perf_counter() - started
        audio_seconds = sum(r.duration_s for r in records)
        row = {
            "condition": condition,
            "clips": len(records),
            "event_f1": metrics["event_based"]["micro"]["f1"],
            "macro_event_f1": metrics["event_based"]["macro_f1"],
            "segment_f1": metrics["segment_based"]["micro"]["f1"],
            "inference_seconds": model_seconds,
            "model_rtf": model_seconds / audio_seconds,
            "evaluation_seconds_including_repeat": elapsed,
            "evaluation_rtf_including_repeat": elapsed / (2 * audio_seconds),
            "prediction_sha256": fingerprint(hashes),
            "repeat_prediction_sha256": fingerprint(repeat_hashes),
            "reproducible": hashes == repeat_hashes,
            "per_class": metrics["event_based"]["per_class"],
        }
        rows.append(row)
        print(
            f"{condition}: event F1={row['event_f1']:.4f}, repeat={row['reproducible']}", flush=True
        )
    clean = rows[0]["event_f1"]
    for row in rows:
        row["event_f1_drop_absolute"] = clean - row["event_f1"]
        row["event_f1_drop_relative"] = (clean - row["event_f1"]) / max(clean, 1e-12)
    result = {
        "status": "passed" if all(r["reproducible"] for r in rows) else "failed",
        "protocol": "controlled-software-domain-shift-v1",
        "seed": 2027,
        "physical_validation_performed": False,
        "test_clips": len(records),
        "model_sha256": _sha256(root / MODEL),
        "onnx_sha256": _sha256(root / ONNX),
        "manifest_sha256": _sha256(manifest),
        "threshold_profile": "overall",
        "notes": "Simulated impulse responses, bandwidth and noise; no physical devices. "
        "Causal transforms keep source annotation times and crop reverb tails. "
        "Peak normalization only when exceeding 0.98; no test-set tuning.",
        "conditions": rows,
    }
    write_json(root / OUT / "robustness.json", result)
    return result


def soak(root, seconds=1800, model_path=MODEL, onnx_path=ONNX, output_dir=OUT):
    deployment = SEDDeployment(root / model_path, root / onnx_path)
    samples, _ = read_wav_mono(
        root / "data/l4-engineering/l4_long_demo.wav", deployment.sample_rate
    )
    count = round(seconds * deployment.sample_rate)
    processor = StreamingSEDProcessor(deployment)
    process = psutil.Process()
    elapsed_chunks, rss, sizes, events = [], [], [], []
    started = time.perf_counter()
    for left in range(0, count, processor.chunk_samples):
        indices = np.arange(left, min(left + processor.chunk_samples, count)) % len(samples)
        begin = time.perf_counter()
        events.extend(processor.process_chunk(samples[indices]))
        elapsed_chunks.append(time.perf_counter() - begin)
        if left >= 30 * deployment.sample_rate:
            rss.append(process.memory_info().rss)
        sizes.append(processor.state_sizes())
    events.extend(processor.flush())
    elapsed = time.perf_counter() - started
    repeat_hashes = []
    for _ in range(2):
        short = StreamingSEDProcessor(deployment)
        result = []
        for start in range(0, min(len(samples), 30 * deployment.sample_rate), short.chunk_samples):
            result.extend(short.process_chunk(samples[start : start + short.chunk_samples]))
        result.extend(short.flush())
        repeat_hashes.append(fingerprint(result))
    keys = [(e["label"], e["onset_s"], e["offset_s"]) for e in events]
    last = {}
    monotonic = True
    for event in events:
        monotonic &= event["onset_s"] >= last.get(event["label"], -1)
        last[event["label"]] = event["offset_s"]
    maxima = {key: max(s[key] for s in sizes) for key in sizes[0]}
    limits = {
        "audio_samples": processor.context_samples,
        "pending_samples": processor.chunk_samples,
        "median_frames": processor.decoder.width,
        "class_states": len(deployment.class_names),
        "latency_entries": 2048,
    }
    growth = max(rss) - min(rss) if rss else 0
    p95 = float(np.percentile(elapsed_chunks, 95))
    checks = {
        "full_duration": abs(processor.processed_seconds - seconds) < 1e-6,
        "compute_p95_below_chunk": p95 < 1,
        "rtf_below_01": elapsed / seconds < 0.1,
        "rss_growth_below_64mib": growth <= 64 * 1024**2,
        "bounded_state": all(maxima[k] <= limits[k] for k in maxima),
        "no_duplicate_events": len(keys) == len(set(keys)),
        "per_class_chronological": bool(monotonic),
        "repeat_event_hash_match": repeat_hashes[0] == repeat_hashes[1],
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "mode": "accelerated_file_stream",
        "physical_validation_performed": False,
        "processed_seconds": processor.processed_seconds,
        "wall_seconds": elapsed,
        "rtf": elapsed / seconds,
        "compute_p95_ms": p95 * 1000,
        "rss_growth_bytes_after_warmup": growth,
        "state_maxima": maxima,
        "state_limits": limits,
        "checks": checks,
        "repeat_event_hashes": repeat_hashes,
        "event_count": len(events),
        "model_sha256": _sha256(root / model_path),
        "onnx_sha256": _sha256(root / onnx_path),
        "note": "1,800 seconds of audio processed without clock pacing; not a 30-minute "
        "wall-clock microphone run. Events are finalized in availability order; "
        "time ordering is enforced per class.",
    }
    write_json(root / output_dir / "stream_soak.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["evaluate", "soak"])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seconds", type=float, default=1800)
    args = parser.parse_args()
    result = (
        evaluate(args.root.resolve())
        if args.command == "evaluate"
        else soak(args.root.resolve(), args.seconds)
    )
    print(json.dumps({k: v for k, v in result.items() if k != "conditions"}, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
