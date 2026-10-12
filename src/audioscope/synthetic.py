from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
from scipy.io import wavfile

from .manifest import AudioRecord, sha256_file, write_manifest


def generate_smoke_dataset(root: Path, sample_rate: int, duration_s: float, seed: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    specs = {"low_tone": 220.0, "mid_tone": 440.0, "high_tone": 880.0}
    records: List[AudioRecord] = []
    sample_count = int(sample_rate * duration_s)
    time = np.arange(sample_count, dtype=np.float64) / sample_rate
    for class_index, (label, frequency) in enumerate(specs.items()):
        class_dir = root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for item_index in range(10):
            item_seed = seed + class_index * 100 + item_index
            rng = np.random.default_rng(item_seed)
            phase = rng.uniform(0.0, 2.0 * math.pi)
            harmonic = 0.1 * np.sin(2.0 * math.pi * frequency * 2.0 * time + phase / 2.0)
            signal = 0.55 * np.sin(2.0 * math.pi * frequency * time + phase) + harmonic
            signal += rng.normal(0.0, 0.005, size=sample_count)
            signal *= np.hanning(sample_count)
            pcm = np.clip(signal * 32767.0, -32768, 32767).astype(np.int16)
            filename = f"{label}-{item_index:02d}.wav"
            audio_path = class_dir / filename
            wavfile.write(audio_path, sample_rate, pcm)
            relative_path = audio_path.relative_to(root).as_posix()
            records.append(
                AudioRecord(
                    asset_id=f"smoke-{label}-{item_index:02d}",
                    path=relative_path,
                    sha256=sha256_file(audio_path),
                    dataset="audioscope-smoke",
                    dataset_version="1.0.0",
                    source_url="generated://audioscope/smoke-v1",
                    license="CC0-1.0",
                    sample_rate=sample_rate,
                    channels=1,
                    duration_s=duration_s,
                    label=label,
                    fold=(item_index % 5) + 1,
                    source_recording_id=f"smoke-{label}-{item_index:02d}",
                )
            )
    manifest_path = root / "manifest.jsonl"
    write_manifest(manifest_path, records)
    return manifest_path
