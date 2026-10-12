from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class AudioRecord:
    asset_id: str
    path: str
    sha256: str
    dataset: str
    dataset_version: str
    source_url: str
    license: str
    sample_rate: int
    channels: int
    duration_s: float
    label: str
    fold: int
    source_recording_id: str

    @classmethod
    def from_dict(cls, value: dict) -> "AudioRecord":
        return cls(**value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> List[AudioRecord]:
    records: List[AudioRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(AudioRecord.from_dict(json.loads(line)))
            except Exception as error:
                raise ValueError(f"Invalid manifest line {line_number}: {error}") from error
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


def write_manifest(path: Path, records: Iterable[AudioRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")


def validate_manifest(
    records: Iterable[AudioRecord],
    data_root: Path,
    verify_hash: bool = True,
    allow_source_fold_overlap: bool = False,
) -> dict:
    records = list(records)
    errors = []
    seen_assets = set()
    source_folds = {}
    for record in records:
        if record.asset_id in seen_assets:
            errors.append(f"duplicate asset_id: {record.asset_id}")
        seen_assets.add(record.asset_id)
        if not record.label or not record.license or not record.sha256:
            errors.append(f"missing label/license/hash: {record.asset_id}")
        if record.fold < 1:
            errors.append(f"invalid fold: {record.asset_id}")
        source_folds.setdefault(record.source_recording_id, set()).add(record.fold)
        audio_path = data_root / record.path
        if not audio_path.is_file():
            errors.append(f"missing audio: {record.path}")
        elif verify_hash and sha256_file(audio_path) != record.sha256:
            errors.append(f"checksum mismatch: {record.path}")
    source_fold_overlaps = sorted(
        source_id for source_id, folds in source_folds.items() if len(folds) > 1
    )
    if source_fold_overlaps and not allow_source_fold_overlap:
        errors.extend(
            f"source leakage across folds: {source_id}" for source_id in source_fold_overlaps
        )
    if errors:
        raise ValueError("Manifest validation failed:\n- " + "\n- ".join(errors))
    return {
        "records": len(records),
        "classes": len({record.label for record in records}),
        "folds": sorted({record.fold for record in records}),
        "unique_sources": len(source_folds),
        "source_fold_overlaps": source_fold_overlaps,
        "source_fold_overlap_allowed": allow_source_fold_overlap,
        "hashes_verified": verify_hash,
    }
