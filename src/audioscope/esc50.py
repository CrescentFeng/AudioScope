from __future__ import annotations

import csv
import subprocess
import uuid
from pathlib import Path
from typing import List

from scipy.io import wavfile

from .manifest import AudioRecord, sha256_file, write_manifest


def _dataset_revision(dataset_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(dataset_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else "unknown"


def prepare_esc50(dataset_root: Path, output: Path) -> Path:
    metadata_path = dataset_root / "meta" / "esc50.csv"
    audio_root = dataset_root / "audio"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"ESC-50 metadata not found: {metadata_path}")
    revision = _dataset_revision(dataset_root)
    records: List[AudioRecord] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            audio_path = audio_root / row["filename"]
            sample_rate, samples = wavfile.read(audio_path, mmap=True)
            channels = 1 if samples.ndim == 1 else int(samples.shape[1])
            duration_s = float(samples.shape[0]) / sample_rate
            source_id = row.get("src_file") or row["filename"].split("-")[0]
            records.append(
                AudioRecord(
                    asset_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"esc50:{row['filename']}")),
                    path=audio_path.relative_to(dataset_root).as_posix(),
                    sha256=sha256_file(audio_path),
                    dataset="ESC-50",
                    dataset_version=revision,
                    source_url=f"https://github.com/karolpiczak/ESC-50/tree/{revision}",
                    license="CC-BY-NC-3.0",
                    sample_rate=int(sample_rate),
                    channels=channels,
                    duration_s=duration_s,
                    label=row["category"],
                    fold=int(row["fold"]),
                    source_recording_id=f"esc50-source-{source_id}",
                )
            )
    write_manifest(output, records)
    return output
