from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from scipy.io import wavfile

from .sed_data import SoundscapeRecord, TimedEvent, write_soundscape_manifest

DESED_CLASSES = [
    "alarm_bell_ringing",
    "blender",
    "cat",
    "dishes",
    "dog",
    "electric_shaver_toothbrush",
    "frying",
    "running_water",
    "speech",
    "vacuum_cleaner",
]

_SHARD_PATTERN = re.compile(r"wds-audio-(train|eval)-\d{6}\.tar$")


def _file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_recording_id(scene_id: str) -> str:
    # DESED filenames use ``Y`` followed by the 11-character YouTube video ID.
    return scene_id[:12] if scene_id.startswith("Y") and len(scene_id) >= 12 else scene_id


def _hash_fraction(source_recording_id: str) -> float:
    value = int.from_bytes(
        hashlib.sha256(source_recording_id.encode("utf-8")).digest()[:8], "big"
    )
    return value / float(2**64)


def _strong_split(
    source_recording_id: str, train_fraction: float, validation_fraction: float
) -> str:
    value = _hash_fraction(source_recording_id)
    if value < train_fraction:
        return "train"
    if value < train_fraction + validation_fraction:
        return "validation"
    return "test"


def _normalise_label(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _read_shard_index(path: Path) -> List[dict]:
    match = _SHARD_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected DESED shard name: {path.name}")
    pool = match.group(1)
    rows = []
    with tarfile.open(path, "r") as archive:
        members = {member.name: member for member in archive.getmembers()}
        for json_name in sorted(name for name in members if name.endswith(".json")):
            scene_id = Path(json_name).stem
            wav_name = f"{scene_id}.wav"
            if wav_name not in members:
                raise ValueError(f"Missing {wav_name} in {path.name}")
            metadata_stream = archive.extractfile(members[json_name])
            if metadata_stream is None:
                raise ValueError(f"Cannot read {json_name} from {path.name}")
            annotations = json.loads(metadata_stream.read().decode("utf-8"))
            events = []
            for annotation in annotations:
                onset = float(annotation["onset"])
                offset = float(annotation["offset"])
                if not 0.0 <= onset < offset:
                    raise ValueError(f"Invalid event interval in {json_name}: {annotation}")
                for raw_label in str(annotation["label"]).split(","):
                    label = _normalise_label(raw_label)
                    if label not in DESED_CLASSES:
                        raise ValueError(f"Unknown DESED label {raw_label!r} in {json_name}")
                    events.append((label, onset, offset))
            rows.append(
                {
                    "pool": pool,
                    "shard": path,
                    "wav_member": members[wav_name],
                    "scene_id": scene_id,
                    "source_recording_id": _source_recording_id(scene_id),
                    "events": sorted(set(events), key=lambda value: (value[1], value[0])),
                }
            )
    return rows


def prepare_desed_real(
    shard_root: Path,
    output_root: Path,
    settings: Mapping[str, Any],
) -> dict:
    """Prepare the real, strongly annotated DESED WebDataset repack.

    The official train pool remains train. The official eval pool is divided by a stable hash of
    the source YouTube recording ID, keeping all segments from one source in the same split.
    """

    shards = sorted(shard_root.glob("wds-audio-*.tar"))
    train_shards = [path for path in shards if "-train-" in path.name]
    eval_shards = [path for path in shards if "-eval-" in path.name]
    expected_train = int(settings.get("expected_train_shards", 20))
    expected_eval = int(settings.get("expected_eval_shards", 20))
    if len(train_shards) != expected_train or len(eval_shards) != expected_eval:
        raise ValueError(
            "Incomplete DESED download: expected "
            f"{expected_train} train/{expected_eval} eval shards, found "
            f"{len(train_shards)} train/{len(eval_shards)} eval"
        )

    shard_md5 = {path.name: _file_md5(path) for path in shards}
    signature_payload = {
        "shard_md5": shard_md5,
        "strong_only": bool(settings.get("strong_only", False)),
        "train_fraction": float(settings.get("train_fraction", 0.0)),
        "validation_fraction": float(settings.get("validation_fraction", 0.5)),
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    metadata_path = output_root / "metadata.json"
    manifest_path = output_root / "manifest.jsonl"
    if metadata_path.is_file() and manifest_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("signature") == signature:
            return existing

    indexed = []
    for index, path in enumerate(shards, 1):
        indexed.extend(_read_shard_index(path))
        print(f"DESED index: {index}/{len(shards)} shards", flush=True)

    train_sources = {
        row["source_recording_id"] for row in indexed if row["pool"] == "train"
    }
    eval_sources = {
        row["source_recording_id"] for row in indexed if row["pool"] == "eval"
    }
    cross_pool_sources = train_sources & eval_sources
    strong_only = bool(settings.get("strong_only", False))
    train_fraction = float(settings.get("train_fraction", 0.0))
    validation_fraction = float(settings.get("validation_fraction", 0.5))
    if strong_only:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between zero and one")
        if not 0.0 < validation_fraction < 1.0 - train_fraction:
            raise ValueError("validation_fraction leaves no non-empty test fraction")
    elif not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")

    output_records = []
    skipped_cross_pool_clips = 0
    excluded_weak_train_clips = 0
    by_shard: Dict[Path, List[Tuple[dict, str]]] = {}
    for row in indexed:
        if strong_only and row["pool"] == "train":
            excluded_weak_train_clips += 1
            continue
        if row["pool"] == "eval" and row["source_recording_id"] in cross_pool_sources:
            skipped_cross_pool_clips += 1
            continue
        if strong_only:
            split = _strong_split(
                row["source_recording_id"], train_fraction, validation_fraction
            )
        else:
            split = (
                "train"
                if row["pool"] == "train"
                else (
                    "validation"
                    if _hash_fraction(row["source_recording_id"]) < validation_fraction
                    else "test"
                )
            )
        by_shard.setdefault(row["shard"], []).append((row, split))

    for shard_index, (shard, rows) in enumerate(sorted(by_shard.items()), 1):
        with tarfile.open(shard, "r") as archive:
            for row, split in rows:
                stream = archive.extractfile(row["wav_member"])
                if stream is None:
                    raise ValueError(
                        f"Cannot read {row['wav_member'].name} from {shard.name}"
                    )
                wav_bytes = stream.read()
                sample_rate, samples = wavfile.read(io.BytesIO(wav_bytes))
                duration_s = len(samples) / float(sample_rate)
                relative_path = Path("audio") / split / f"{row['scene_id']}.wav"
                destination = output_root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(wav_bytes)
                events = tuple(
                    TimedEvent(
                        label=label,
                        onset_s=onset,
                        offset_s=min(offset, duration_s),
                        source_asset_id=row["scene_id"],
                        source_recording_id=row["source_recording_id"],
                    )
                    for label, onset, offset in row["events"]
                    if onset < duration_s
                )
                output_records.append(
                    SoundscapeRecord(
                        scene_id=row["scene_id"],
                        path=relative_path.as_posix(),
                        split=split,
                        duration_s=duration_s,
                        sample_rate=int(sample_rate),
                        events=events,
                    )
                )
        print(f"DESED extract: {shard_index}/{len(by_shard)} shards", flush=True)

    output_records.sort(key=lambda record: (record.split, record.scene_id))
    write_soundscape_manifest(manifest_path, output_records)
    split_names = ("train", "validation", "test")
    source_sets = {
        split: {
            event.source_recording_id
            for record in output_records
            if record.split == split
            for event in record.events
        }
        for split in split_names
    }
    overlap_checks = {
        "train_validation": sorted(source_sets["train"] & source_sets["validation"]),
        "train_test": sorted(source_sets["train"] & source_sets["test"]),
        "validation_test": sorted(source_sets["validation"] & source_sets["test"]),
    }
    if any(overlap_checks.values()):
        raise RuntimeError(f"DESED source leakage after preparation: {overlap_checks}")

    split_stats = {}
    for split in split_names:
        split_records = [record for record in output_records if record.split == split]
        class_counts = Counter(
            event.label for record in split_records for event in record.events
        )
        missing = sorted(set(DESED_CLASSES) - set(class_counts))
        if missing:
            raise ValueError(f"DESED split {split!r} has no events for classes: {missing}")
        split_stats[split] = {
            "clips": len(split_records),
            "events": sum(len(record.events) for record in split_records),
            "unique_source_recordings": len(source_sets[split]),
            "short_events_le_1s": sum(
                event.offset_s - event.onset_s <= 1.0
                for record in split_records
                for event in record.events
            ),
            "class_events": dict(sorted(class_counts.items())),
        }

    metadata = {
        "dataset": "DESED real strong - WebDataset repack v2",
        "source_record": "https://doi.org/10.5281/zenodo.14808180",
        "license": "CC-BY-4.0 as declared by the Zenodo repack",
        "signature": signature,
        "shard_md5": shard_md5,
        "train_shards": len(train_shards),
        "eval_shards": len(eval_shards),
        "classes": DESED_CLASSES,
        "split_protocol": (
            "strong eval pool only; stable SHA-256 source-ID hash into train/validation/test"
            if strong_only
            else "official train pool plus hashed eval validation/test"
        ),
        "strong_only": strong_only,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": 1.0 - train_fraction - validation_fraction,
        "excluded_weak_train_clips": excluded_weak_train_clips,
        "excluded_cross_pool_source_ids": sorted(cross_pool_sources),
        "skipped_cross_pool_clips": skipped_cross_pool_clips,
        "source_overlap_checks": overlap_checks,
        "splits": split_stats,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata
