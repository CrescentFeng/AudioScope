from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from audioscope.manifest import AudioRecord
from audioscope.sed import CRNN, LiteConformer, _loss_weights
from audioscope.sed_data import generate_esc10_soundscapes, read_soundscape_manifest


def test_crnn_preserves_time_axis():
    model = CRNN(n_mels=32, num_classes=3, channels=[4, 8], gru_hidden_size=8, dropout=0.1)
    output = model(torch.zeros(2, 1, 32, 40))
    assert output.shape == (2, 40, 3)


def test_lite_conformer_restores_time_axis_after_pooling():
    model = LiteConformer(
        n_mels=32,
        num_classes=3,
        channels=[4, 8],
        dimension=16,
        heads=4,
        layers=1,
        feedforward=32,
        convolution_kernel=7,
        dropout=0.1,
        time_pool_factors=[2, 1],
    )
    output = model(torch.zeros(2, 1, 32, 40))
    assert output.shape == (2, 40, 3)


def test_short_event_loss_weighting_emphasises_short_runs_and_masks_padding():
    targets = np.zeros((1, 20, 1), dtype=np.uint8)
    targets[0, 2:5, 0] = 1
    targets[0, 8:18, 0] = 1
    valid = np.zeros((1, 20), dtype=bool)
    valid[:, :18] = True
    weights = _loss_weights(
        targets,
        valid,
        hop_s=0.1,
        settings={
            "enabled": True,
            "maximum_duration_s": 0.5,
            "positive_weight": 3.0,
            "boundary_s": 0.0,
            "boundary_weight": 1.0,
        },
    )
    assert np.all(weights[0, 2:5, 0] == 3.0)
    assert np.all(weights[0, 8:18, 0] == 1.0)
    assert np.all(weights[0, 18:, 0] == 0.0)


def test_soundscape_generator_keeps_source_splits_disjoint(tmp_path: Path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    records = []
    for fold in (1, 2, 3):
        for label_index, label in enumerate(("dog", "rain")):
            samples = np.sin(
                2 * np.pi * (200 + label_index * 100) * np.arange(8000) / 8000
            ).astype(np.float32)
            filename = f"{fold}-{label}.wav"
            wavfile.write(source_root / filename, 8000, samples)
            records.append(
                AudioRecord(
                    asset_id=f"asset-{fold}-{label}",
                    path=filename,
                    sha256=f"hash-{fold}-{label}",
                    dataset="test",
                    dataset_version="1",
                    source_url="local",
                    license="CC0-1.0",
                    sample_rate=8000,
                    channels=1,
                    duration_s=1.0,
                    label=label,
                    fold=fold,
                    source_recording_id=f"source-{fold}-{label}",
                )
            )
    output_root = tmp_path / "soundscapes"
    metadata = generate_esc10_soundscapes(
        records,
        source_root,
        output_root,
        {
            "sample_rate": 8000,
            "duration_s": 1.0,
            "scenes": {"train": 2, "validation": 2, "test": 2},
            "split_folds": {"train": [1], "validation": [2], "test": [3]},
            "classes": ["dog", "rain"],
            "events_per_scene": [2, 2],
            "event_duration_s": [0.2, 0.4],
            "overlap_probability": 1.0,
        },
        seed=2027,
    )
    scenes = read_soundscape_manifest(output_root / "manifest.jsonl")
    assert len(scenes) == 6
    assert not metadata["source_overlap_checks"]["train_test"]
    assert all((output_root / scene.path).is_file() for scene in scenes)
