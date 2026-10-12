"""Frozen external evaluation. Official labels are never generated from predictions."""
import argparse
import csv
import hashlib
import json
import math
import tarfile
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, resample_poly, spectrogram

from audioscope.audio import read_wav_mono

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external-desed-public"
OUT = ROOT / "reports/generated/external-desed-public"
SOURCE = ROOT / "data/external-strong-source-audit"
ANNOTATION_SHA = "da3f978674311b6a7a86224a58ebab9e45d66c1d0b32ad3bd6c126cb4af76f5d"
ARCHIVE_MD5 = "c1bf328257523f7e93244f3c1b160eb4"
EVENT_SETTINGS = {"onset_collar_s": .2, "offset_collar_s": .2, "offset_ratio": .2}


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def pcm_hash(samples):
    # Float32 mono/resampled representation, without extra normalization/clipping.
    return hashlib.sha256(np.asarray(samples, dtype="<f4").tobytes()).hexdigest()


def lowband(samples):
    return resample_poly(samples, 1, 8).astype(np.float32)


@lru_cache(maxsize=256)
def historical_lowband(path):
    samples, _ = read_wav_mono(path, 16000)
    return lowband(samples)


def fingerprints(x):
    """3 s windows at 1 s strides; log spectra, invariant to polarity and gain."""
    result, starts = [], []
    for start in range(0, len(x) - 6000 + 1, 2000):
        window = x[start:start + 6000]
        if np.std(window) < 1e-5:
            continue
        _, _, power = spectrogram(window, fs=2000, nperseg=256, noverlap=128)
        bands = np.array([p.mean(axis=0) for p in np.array_split(power[1:], 16)])
        features = np.log(np.maximum(bands, 1e-14))
        features = np.concatenate([p.mean(axis=1) for p in np.array_split(features, 6, axis=1)])
        features -= features.mean()
        norm = np.linalg.norm(features)
        if norm > 1e-8:
            result.append(features / norm)
            starts.append(start)
    return np.asarray(result, dtype=np.float32).reshape(-1, 96), starts


def max_correlation(template, recording):
    """Absolute normalized cross-correlation with all valid sample offsets."""
    if len(recording) < len(template) or np.std(template) < 1e-5:
        return 0.0
    template = np.array(template, dtype=np.float64, copy=True)
    recording = np.asarray(recording, dtype=np.float64)
    template -= template.mean()
    n = len(template)
    sums = np.r_[0., np.cumsum(recording)]
    squares = np.r_[0., np.cumsum(recording ** 2)]
    energy = squares[n:] - squares[:-n] - (sums[n:] - sums[:-n]) ** 2 / n
    denominator = np.sqrt(np.maximum(energy, 0.) * np.sum(template ** 2))
    values = np.divide(np.abs(correlate(recording, template, mode="valid", method="fft")),
                       denominator, out=np.zeros_like(denominator), where=denominator > 1e-10)
    return float(min(1., values.max(initial=0.)))


def build_history():
    # Include original corpora, generated scenes, smoke/field/pilot audio; all splits.
    paths = sorted(p for p in (ROOT / "data").rglob("*.wav") if DATA not in p.parents)
    items, seen, vectors, owners = [], {}, [], []
    for i, path in enumerate(paths):
        file_hash = digest(path)
        if file_hash in seen:
            items[seen[file_hash]]["aliases"].append(str(path.relative_to(ROOT)))
            continue
        x, _ = read_wav_mono(path, 16000)
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite historical audio: {path}")
        index = len(items)
        seen[file_hash] = index
        items.append({"path": str(path.relative_to(ROOT)), "aliases": [],
                      "sha256": file_hash, "pcm_sha256": pcm_hash(x), "samples": len(x)})
        fp, _ = fingerprints(lowband(x))
        vectors.extend(fp)
        owners.extend([index] * len(fp))
        if i % 500 == 0:
            print(f"history {i}/{len(paths)}", flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA / "history-fingerprints.npz", vectors=np.asarray(vectors), owners=owners)
    write(DATA / "history-index.json", {"scanned_wav_paths": len(paths), "unique_file_hashes": len(items),
                                       "items": items, "scope": "All pre-existing WAV files under project data/"})
    print(f"history complete: {len(paths)} paths, {len(items)} unique files", flush=True)


def validate_events(events, duration):
    for e in events:
        a, b = e["onset_s"], e["offset_s"]
        if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= duration):
            raise ValueError(f"Annotation outside actual audio: {a}, {b}, duration={duration}")


def extract_verified(archive, target):
    if archive.stat().st_size != 889698336 or digest(archive, "md5") != ARCHIVE_MD5:
        raise ValueError("Official archive size/MD5 mismatch")
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        for m in tf.getmembers():
            destination = (target / m.name).resolve()
            if not destination.is_relative_to(target.resolve()) or not (m.isfile() or m.isdir()):
                raise ValueError(f"Unsafe archive member: {m.name}")
        tf.extractall(target, filter="data")


def annotation_rows(path):
    with path.open() as f:
        return sorted((r["filename"], float(r["onset"]), float(r["offset"]), r["event_label"].lower())
                      for r in csv.DictReader(f, delimiter="\t"))


def license_eligible(row):
    return bool(row and all(row.get(k, "").strip() for k in ("filename", "uploader", "uploader_id"))
                and row.get("license") == "Creative Commons Attribution license (reuse allowed)")


def prepare():
    annotation = SOURCE / "desed-public-eval.official.tsv"
    if digest(annotation) != ANNOTATION_SHA:
        raise ValueError("Pinned annotation changed")
    archive = DATA / "downloads/DESED_public_eval.tar.gz.part"
    extracted = DATA / "official"
    extract_verified(archive, extracted)
    official = annotation_rows(annotation)
    bundled = []
    for p in extracted.rglob("*.tsv"):
        try:
            rows = annotation_rows(p)
        except (KeyError, ValueError):
            continue
        bundled.append({"path": str(p.relative_to(DATA)), "sha256": digest(p),
                        "same_as_pinned": rows == official, "events": len(rows)})
    if any(not r["same_as_pinned"] for r in bundled):
        raise ValueError(f"Archive annotation revision differs: {bundled}")
    license_file = extracted / "license_public_eval.tsv"
    with license_file.open() as handle:
        license_rows = list(csv.DictReader(handle, delimiter="\t"))
    licenses = {r["filename"]: r for r in license_rows}
    if len(licenses) != len(license_rows):
        raise ValueError("Ambiguous duplicate per-recording license entries")
    history = json.loads((DATA / "history-index.json").read_text())
    h = history["items"]
    # Do not silently reuse a stale history index.
    for item in h:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Historical audio changed after indexing")
    index = np.load(DATA / "history-fingerprints.npz")
    vectors, owners = index["vectors"], index["owners"]
    old_file = {r["sha256"]: r["path"] for r in h}
    old_pcm = {r["pcm_sha256"]: r["path"] for r in h}
    candidates = [json.loads(s) for s in (SOURCE / "candidate-manifest.jsonl").read_text().splitlines()]
    groups = defaultdict(list)
    for name, onset, offset, label in official:
        groups[name].append({"label": label, "onset_s": onset, "offset_s": offset})
    all_audio = {p.name: p for p in extracted.rglob("*.wav")}
    if set(groups) != set(all_audio):
        raise ValueError(f"Audio/annotation file coverage mismatch: {len(groups)}, {len(all_audio)}")
    retained, exclusions, integrity, seen_pcm = [], [], [], {}
    for i, candidate in enumerate(candidates):
        name = candidate["filename"]
        path = all_audio[name]
        rate, raw = wavfile.read(path)
        if rate <= 0 or raw.size == 0 or not np.isfinite(raw).all():
            raise ValueError(f"Invalid original PCM: {name}")
        if raw.dtype != np.int16:
            raise ValueError(f"Unexpected official PCM format, review before conversion: {name}: {raw.dtype}")
        duration = len(raw) / rate
        events = groups[name]
        # Original annotation is immutable. Invalid boundaries are quarantined, not clipped.
        try:
            validate_events(events, duration)
        except ValueError as exc:
            exclusions.append({"filename": name, "reason": "invalid_time_boundary", "detail": str(exc)})
            continue
        x, _ = read_wav_mono(path, 16000)
        if not np.isfinite(x).all() or len(x) == 0:
            raise ValueError(f"Invalid audio: {name}")
        file_hash, canonical = digest(path), pcm_hash(x)
        matches = []
        if file_hash in old_file or canonical in old_pcm:
            matches.append({"path": old_file.get(file_hash, old_pcm.get(canonical)), "method": "exact_file_or_pcm"})
        if canonical in seen_pcm:
            matches.append({"path": seen_pcm[canonical], "method": "within_external_exact_pcm"})
        y = lowband(x)
        fps, starts = fingerprints(y)
        tested = set()
        # Candidate retrieval is a screen, not a proof of absence of all near-duplicates.
        for fp, start in zip(fps, starts):
            scores = vectors @ fp
            top = np.argpartition(scores, -min(10, len(scores)))[-10:]
            for owner in set(owners[top].tolist()):
                if (owner, start) in tested:
                    continue
                tested.add((owner, start))
                correlation = max_correlation(y[start:start + 6000], historical_lowband(ROOT / h[owner]["path"]))
                if correlation >= .98:
                    matches.append({"path": h[owner]["path"], "method": "near_duplicate_screen",
                                    "absolute_ncc": correlation, "query_onset_s": start / 2000})
        integrity.append({"filename": name, "sha256": file_hash, "pcm_sha256": canonical,
                          "sample_rate": rate, "channels": 1 if raw.ndim == 1 else raw.shape[1],
                          "duration_s": duration, "resampled_peak": float(np.max(np.abs(x))),
                          "candidate_alignment_checks": len(tested)})
        if matches:
            exclusions.append({"filename": name, "reason": "conservative_waveform_duplicate_quarantine", "matches": matches})
            continue
        if not license_eligible(licenses.get(name)):
            exclusions.append({"filename": name, "reason": "incomplete_per_recording_license_or_attribution",
                               "upstream_entry": licenses.get(name)})
            continue
        seen_pcm[canonical] = name
        scene_id = Path(name).stem
        retained.append({"scene_id": scene_id, "path": str(path.relative_to(DATA)), "split": "external_test",
                         "sample_rate": rate, "duration_s": duration,
                         "events": [dict(e, source_asset_id=scene_id,
                                         source_recording_id=candidate["source_recording_id"]) for e in events]})
        write(DATA / "references" / f"{scene_id}.reference.json",
              {"source": "dataset_reference", "complete": True, "audio_sha256": file_hash,
               "annotation_source": "DESED official human strong annotations", "annotation_sha256": ANNOTATION_SHA,
               "scope": "Ten known classes only; not an exhaustive ontology of every audible sound",
               "upstream_attribution": licenses[name], "formal_evaluation_eligible": True,
               "events": events, "assistant_listened": False, "training_eligible": False})
        if i % 25 == 0:
            print(f"integrity/dedup {i}/{len(candidates)} retained={len(retained)}", flush=True)
    DATA.joinpath("manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in retained))
    write(OUT / "data-audit.json", {
        "status": "completed", "official_record": "https://zenodo.org/records/4560759",
        "archive_md5": ARCHIVE_MD5, "archive_sha256": digest(archive), "annotation_sha256": ANNOTATION_SHA,
        "bundled_annotation_comparison": bundled, "official_recordings": len(groups),
        "license_table_sha256": digest(license_file),
        "license_table_rows": len(license_rows),
        "license_eligible_official_rows": sum(license_eligible(r) for r in license_rows),
        "license_policy": "Dataset record says CC-BY-4.0; only clips with explicit upstream per-recording CC attribution and uploader/id enter this evaluation. No legal certification or public redistribution.",
        "source_id_exclusions": json.loads((SOURCE / "source-overlap-audit.json").read_text())["excluded"],
        "additional_exclusions": exclusions, "retained_recordings": len(retained),
        "retained_events": sum(len(r["events"]) for r in retained),
        "class_events": dict(Counter(e["label"] for r in retained for e in r["events"])),
        "history_paths_checked": history["scanned_wav_paths"], "history_unique_files": len(h),
        "history_index_sha256": digest(DATA / "history-index.json"),
        "history_fingerprints_sha256": digest(DATA / "history-fingerprints.npz"),
        "waveform_screen": "Exact file and mono16k float32 PCM SHA256; 3s/1s log-spectral top10 retrieval, 2kHz absolute NCC>=0.98 quarantine",
        "duplicate_limitations": "Approximate top10 retrieval is not exhaustive; overlaps under 3s, transformed mixes or time stretching can escape. Pretraining overlap unknown.",
        "physical_validation_performed": False, "audio_integrity": integrity})


def evaluate():
    import torch

    from audioscope.engineering import SEDDeployment
    from audioscope.event_review import error_triage
    from audioscope.sed_data import TimedEvent, read_soundscape_manifest
    from audioscope.sed_metrics import duration_binned_event_metrics, evaluate_sed
    torch.set_num_threads(4)
    registry_path = ROOT / "artifacts/workbench-release/model_registry.json"
    registry = json.loads(registry_path.read_text())
    manifest = DATA / "manifest.jsonl"
    records = read_soundscape_manifest(manifest)
    audit = json.loads((OUT / "data-audit.json").read_text())
    if audit["status"] != "completed" or len(records) != audit["retained_recordings"]:
        raise ValueError("Data audit incomplete")
    key = registry["default"]
    model = registry["models"][key]
    for field in ("checkpoint", "onnx"):
        if digest(ROOT / model[field]) != model[f"{field}_sha256"]:
            raise ValueError("Frozen model hash changed")
    protocol = {"model": key, "model_details": model, "registry_sha256": digest(registry_path),
                "manifest_sha256": digest(manifest), "data_audit_sha256": digest(OUT / "data-audit.json"),
                "code_sha256": digest(Path(__file__)), "event_settings": EVENT_SETTINGS,
                "segment_resolution_s": 1., "threshold_profile": "overall", "seed": 2027,
                "test_only": True, "training_or_threshold_tuning": False, "physical_validation_performed": False}
    protocol_file = OUT / "frozen-protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError("Cannot silently change frozen external protocol")
    write(protocol_file, protocol)
    deployment = SEDDeployment(ROOT / model["checkpoint"], ROOT / model["onnx"])
    if set(audit["class_events"]) - set(deployment.class_names):
        raise ValueError("Official reference class is outside frozen model ontology")
    estimates, cases, seconds = {}, [], []
    for i, record in enumerate(records):
        x, _ = read_wav_mono(DATA / record.path, 16000)
        start = time.perf_counter()
        result = deployment.predict_samples(x, include_unknown=False)
        seconds.append(time.perf_counter() - start)
        estimates[record.scene_id] = [TimedEvent(e["label"], e["onset_s"], e["offset_s"], "prediction", "prediction")
                                      for e in result["events"]]
        ref = [{"label": e.label, "onset_s": e.onset_s, "offset_s": e.offset_s} for e in record.events]
        cases.append({"scene_id": record.scene_id, "audio_path": record.path,
                      "reference": ref, "predictions": result["events"], **error_triage(ref, result["events"])})
        if i % 50 == 0:
            print(f"frozen evaluation {i}/{len(records)}", flush=True)
    metrics = evaluate_sed(records, estimates, deployment.class_names, EVENT_SETTINGS, 1.)
    report = {"status": "completed", "protocol_sha256": digest(protocol_file), "model": key,
              "clips": len(records), "audio_seconds": sum(r.duration_s for r in records),
              "reference_events": sum(len(r.events) for r in records), "metrics": metrics,
              "duration_metrics": duration_binned_event_metrics(records, estimates, deployment.class_names, EVENT_SETTINGS, 1.),
              "inference_seconds": sum(seconds), "rtf": sum(seconds) / sum(r.duration_s for r in records),
              "clip_compute_p95_ms": float(np.percentile(seconds, 95) * 1000),
              "fn_categories": dict(Counter(e["category"] for c in cases for e in c["missed_reference"])),
              "fp_categories": dict(Counter(e["category"] for c in cases for e in c["extra_prediction"])),
              "physical_validation_performed": False,
              "limitations": ["External held-out subset of the same DESED family, not an independent dataset family.",
                              "Official human annotations are a reference, not infallible truth; no independent listening performed.",
                              "No self-owned physical device or room certification.", "Pretraining overlap unknown.",
                              "No threshold tuning, training, or model promotion on these results."]}
    write(OUT / "evaluation.json", report)
    write(OUT / "event-comparisons.json", cases)
    write(OUT / "failure-examples.json", sorted(cases, key=lambda c: c["fn"] + c["fp"], reverse=True)[:30])
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("history", "prepare", "evaluate"))
    stage = parser.parse_args().stage
    {"history": build_history, "prepare": prepare, "evaluate": evaluate}[stage]()
