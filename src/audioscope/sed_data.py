from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.io import wavfile

from .audio import read_wav_mono
from .features import extract_logmel
from .manifest import AudioRecord

ESC10_CLASSES = [
    "chainsaw",
    "clock_tick",
    "crackling_fire",
    "crying_baby",
    "dog",
    "helicopter",
    "rain",
    "rooster",
    "sea_waves",
    "sneezing",
]


@dataclass(frozen=True)
class TimedEvent:
    label: str
    onset_s: float
    offset_s: float
    source_asset_id: str
    source_recording_id: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TimedEvent":
        return cls(**value)


@dataclass(frozen=True)
class SoundscapeRecord:
    scene_id: str
    path: str
    split: str
    duration_s: float
    sample_rate: int
    events: Tuple[TimedEvent, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SoundscapeRecord":
        return cls(
            scene_id=str(value["scene_id"]),
            path=str(value["path"]),
            split=str(value["split"]),
            duration_s=float(value["duration_s"]),
            sample_rate=int(value["sample_rate"]),
            events=tuple(TimedEvent.from_dict(item) for item in value["events"]),
        )


def read_soundscape_manifest(path: Path) -> List[SoundscapeRecord]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(SoundscapeRecord.from_dict(json.loads(line)))
            except Exception as error:
                raise ValueError(
                    f"Invalid soundscape manifest line {line_number}: {error}"
                ) from error
    if not records:
        raise ValueError(f"Soundscape manifest is empty: {path}")
    return records


def write_soundscape_manifest(path: Path, records: Iterable[SoundscapeRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")


def _split_for_fold(fold: int, fold_map: Mapping[str, Sequence[int]]) -> str | None:
    for split, folds in fold_map.items():
        if fold in {int(value) for value in folds}:
            return split
    return None


def _cross_split_sources(
    records: Sequence[AudioRecord], fold_map: Mapping[str, Sequence[int]]
) -> List[str]:
    source_splits: Dict[str, set] = {}
    for record in records:
        split = _split_for_fold(record.fold, fold_map)
        if split is not None:
            source_splits.setdefault(record.source_recording_id, set()).add(split)
    return sorted(source for source, splits in source_splits.items() if len(splits) > 1)


def _event_waveform(
    record: AudioRecord,
    source_root: Path,
    sample_rate: int,
    duration_s: float,
    target_dbfs: float,
    rng: np.random.Generator,
) -> np.ndarray:
    samples, _ = read_wav_mono(source_root / record.path, sample_rate)
    length = min(len(samples), max(1, round(duration_s * sample_rate)))
    start = int(rng.integers(0, max(1, len(samples) - length + 1)))
    clip = samples[start : start + length].copy()
    rms = float(np.sqrt(np.mean(np.square(clip), dtype=np.float64)))
    target_rms = 10.0 ** (target_dbfs / 20.0)
    if rms > 1e-8:
        clip *= target_rms / rms
    fade = min(round(0.02 * sample_rate), len(clip) // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        clip[:fade] *= ramp
        clip[-fade:] *= ramp[::-1]
    return clip.astype(np.float32)


def _scene_labels(
    scene_index: int,
    event_count: int,
    class_names: Sequence[str],
    rng: np.random.Generator,
) -> List[str]:
    primary = class_names[scene_index % len(class_names)]
    candidates = [label for label in class_names if label != primary]
    others = rng.choice(candidates, size=event_count - 1, replace=False).tolist()
    labels = [primary, *others]
    rng.shuffle(labels)
    return labels


def _event_onset(
    events: Sequence[TimedEvent],
    event_duration: float,
    scene_duration: float,
    overlap_probability: float,
    rng: np.random.Generator,
) -> float:
    latest = max(0.0, scene_duration - event_duration)
    if events and rng.random() < overlap_probability:
        anchor = events[-1]
        low = max(0.0, anchor.onset_s - event_duration * 0.6)
        high = min(latest, anchor.offset_s - min(0.15, event_duration * 0.1))
        if high > low:
            return float(rng.uniform(low, high))
    return float(rng.uniform(0.0, latest))


def generate_esc10_soundscapes(
    source_records: Sequence[AudioRecord],
    source_root: Path,
    output_root: Path,
    settings: Mapping[str, Any],
    seed: int,
) -> dict:
    class_names = [str(label) for label in settings.get("classes", ESC10_CLASSES)]
    fold_map = settings["split_folds"]
    excluded_sources = set(_cross_split_sources(source_records, fold_map))
    eligible = [
        record
        for record in source_records
        if record.label in class_names
        and record.source_recording_id not in excluded_sources
        and _split_for_fold(record.fold, fold_map) is not None
    ]
    by_split_label: Dict[str, Dict[str, List[AudioRecord]]] = {}
    for record in eligible:
        split = _split_for_fold(record.fold, fold_map)
        if split is not None:
            by_split_label.setdefault(split, {}).setdefault(record.label, []).append(record)
    for split in fold_map:
        missing = [label for label in class_names if not by_split_label.get(split, {}).get(label)]
        if missing:
            raise ValueError(f"Split {split!r} has no source recordings for classes: {missing}")

    sample_rate = int(settings["sample_rate"])
    scene_duration = float(settings["duration_s"])
    scene_samples = round(sample_rate * scene_duration)
    min_events = int(settings["events_per_scene"][0])
    max_events = int(settings["events_per_scene"][1])
    min_event_s = float(settings["event_duration_s"][0])
    max_event_s = float(settings["event_duration_s"][1])
    overlap_probability = float(settings.get("overlap_probability", 0.5))
    output_records = []
    split_stats = {}
    for split_index, split in enumerate(("train", "validation", "test")):
        count = int(settings["scenes"][split])
        rng = np.random.default_rng(seed + split_index * 100_000)
        split_events = 0
        overlapping_scenes = 0
        split_source_ids = set()
        audio_dir = output_root / "audio" / split
        audio_dir.mkdir(parents=True, exist_ok=True)
        for scene_index in range(count):
            event_count = int(rng.integers(min_events, max_events + 1))
            labels = _scene_labels(scene_index, event_count, class_names, rng)
            mixture = rng.normal(0.0, 10.0 ** (-48.0 / 20.0), scene_samples).astype(np.float32)
            events: List[TimedEvent] = []
            has_overlap = False
            for label in labels:
                record = rng.choice(by_split_label[split][label])
                duration = float(rng.uniform(min_event_s, max_event_s))
                onset = _event_onset(
                    events, duration, scene_duration, overlap_probability, rng
                )
                waveform = _event_waveform(
                    record,
                    source_root,
                    sample_rate,
                    duration,
                    float(rng.uniform(-24.0, -15.0)),
                    rng,
                )
                offset = min(scene_duration, onset + len(waveform) / sample_rate)
                start = round(onset * sample_rate)
                end = min(scene_samples, start + len(waveform))
                mixture[start:end] += waveform[: end - start]
                for previous in events:
                    if onset < previous.offset_s and offset > previous.onset_s:
                        has_overlap = True
                events.append(
                    TimedEvent(
                        label=label,
                        onset_s=round(onset, 6),
                        offset_s=round(offset, 6),
                        source_asset_id=record.asset_id,
                        source_recording_id=record.source_recording_id,
                    )
                )
                split_source_ids.add(record.source_recording_id)
            peak = float(np.max(np.abs(mixture)))
            if peak > 0.98:
                mixture *= 0.98 / peak
            scene_id = f"{split}-{scene_index:04d}"
            relative_path = Path("audio") / split / f"{scene_id}.wav"
            wavfile.write(
                output_root / relative_path,
                sample_rate,
                np.round(np.clip(mixture, -1.0, 1.0) * 32767.0).astype(np.int16),
            )
            ordered_events = tuple(sorted(events, key=lambda event: event.onset_s))
            output_records.append(
                SoundscapeRecord(
                    scene_id=scene_id,
                    path=relative_path.as_posix(),
                    split=split,
                    duration_s=scene_duration,
                    sample_rate=sample_rate,
                    events=ordered_events,
                )
            )
            split_events += len(events)
            overlapping_scenes += int(has_overlap)
        split_stats[split] = {
            "scenes": count,
            "events": split_events,
            "unique_source_recordings": len(split_source_ids),
            "overlapping_scenes": overlapping_scenes,
            "overlapping_scene_fraction": overlapping_scenes / count,
        }

    manifest_path = output_root / "manifest.jsonl"
    write_soundscape_manifest(manifest_path, output_records)
    source_sets = {
        split: {
            event.source_recording_id
            for scene in output_records
            if scene.split == split
            for event in scene.events
        }
        for split in ("train", "validation", "test")
    }
    overlap_checks = {
        "train_validation": sorted(source_sets["train"] & source_sets["validation"]),
        "train_test": sorted(source_sets["train"] & source_sets["test"]),
        "validation_test": sorted(source_sets["validation"] & source_sets["test"]),
    }
    if any(overlap_checks.values()):
        raise RuntimeError(f"Source leakage detected after generation: {overlap_checks}")
    metadata = {
        "dataset": "AudioScope Polyphonic ESC-10 v1",
        "license": "CC-BY-NC-3.0 inherited from ESC-50",
        "seed": seed,
        "classes": class_names,
        "sample_rate": sample_rate,
        "duration_s": scene_duration,
        "source_manifest_records": len(source_records),
        "eligible_source_records": len(eligible),
        "excluded_cross_split_source_recording_ids": sorted(excluded_sources),
        "source_overlap_checks": overlap_checks,
        "settings": dict(settings),
        "splits": split_stats,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def _feature_signature(
    records: Sequence[SoundscapeRecord], feature_settings: Mapping[str, Any]
) -> str:
    digest = hashlib.sha256(json.dumps(dict(feature_settings), sort_keys=True).encode("utf-8"))
    for record in records:
        digest.update(record.scene_id.encode("utf-8"))
        for event in record.events:
            digest.update(
                f"{event.label}:{event.onset_s:.6f}:{event.offset_s:.6f}".encode("utf-8")
            )
    return digest.hexdigest()


def build_sed_tensors(
    records: Sequence[SoundscapeRecord],
    data_root: Path,
    class_names: Sequence[str],
    feature_settings: Mapping[str, Any],
    cache_path: Path,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    signature = _feature_signature(records, feature_settings)
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            cached = np.load(cache_path, allow_pickle=False)
            return cached["features"], cached["targets"], metadata

    class_to_index = {label: index for index, label in enumerate(class_names)}
    target_frames = int(feature_settings["target_frames"])
    hop_s = float(feature_settings["hop_ms"]) / 1000.0
    frame_times = (np.arange(target_frames) + 0.5) * hop_s
    features = []
    targets = []
    for index, record in enumerate(records):
        samples, _ = read_wav_mono(data_root / record.path, record.sample_rate)
        features.append(extract_logmel(samples, record.sample_rate, feature_settings))
        target = np.zeros((target_frames, len(class_names)), dtype=np.uint8)
        for event in record.events:
            active = (frame_times >= event.onset_s) & (frame_times < event.offset_s)
            target[active, class_to_index[event.label]] = 1
        targets.append(target)
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(records):
            print(f"SED features: {index + 1}/{len(records)}", flush=True)
    feature_array = np.stack(features).astype(np.float32)
    target_array = np.stack(targets).astype(np.uint8)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=feature_array, targets=target_array)
    metadata = {
        "signature": signature,
        "records": len(records),
        "feature_shape": list(feature_array.shape),
        "target_shape": list(target_array.shape),
        "hop_s": hop_s,
        "class_names": list(class_names),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return feature_array, target_array, metadata


def count_polyphony(record: SoundscapeRecord, resolution_s: float = 0.02) -> int:
    frames = int(math.ceil(record.duration_s / resolution_s))
    active = np.zeros(frames, dtype=np.int16)
    for event in record.events:
        start = max(0, int(math.floor(event.onset_s / resolution_s)))
        end = min(frames, int(math.ceil(event.offset_s / resolution_s)))
        active[start:end] += 1
    return int(active.max(initial=0))
