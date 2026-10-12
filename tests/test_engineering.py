from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from audioscope.engineering import (
    SEDDeployment,
    StreamingSEDProcessor,
    overlap_add_probabilities,
    overlap_weights,
    window_starts,
)
from audioscope.sed import CRNN


def _artifact(path: Path) -> Path:
    feature_settings = {
        "n_mels": 16,
        "n_fft": 256,
        "frame_ms": 25,
        "hop_ms": 20,
        "target_frames": 50,
    }
    model_settings = {
        "family": "crnn",
        "channels": [4, 8],
        "time_pool_factors": [1, 1],
        "gru_hidden_size": 8,
        "dropout": 0.0,
        "torch_threads": 1,
    }
    model = CRNN(
        n_mels=16,
        num_classes=2,
        channels=[4, 8],
        gru_hidden_size=8,
        dropout=0.0,
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": ["one", "two"],
            "normalization": (0.0, 1.0),
            "feature_settings": feature_settings,
            "model_settings": model_settings,
            "postprocess_settings": {
                "smooth_frames": 1,
                "merge_gap_s": 0.1,
                "minimum_duration_s": 0.04,
            },
            "threshold": 0.999,
            "threshold_profiles": {
                "overall": 0.999,
                "short_event": 0.999,
                "class_balanced": {"one": 0.999, "two": 0.999},
            },
            "sample_rate": 8000,
        },
        path,
    )
    return path


def test_window_starts_cover_short_and_tail_audio():
    assert window_starts(80, 100, 50) == [0]
    starts = window_starts(230, 100, 50)
    assert starts == [0, 50, 100, 150]
    assert starts[-1] + 100 >= 230


def test_overlap_weights_keep_edges_and_peak_in_centre():
    weights = overlap_weights(5)
    assert weights[0] > 0
    assert weights[-1] > 0
    assert weights[2] == 1.0


def test_overlap_add_averages_seam_without_uncovered_frames():
    left = np.zeros((6, 2), dtype=np.float32)
    right = np.ones((6, 2), dtype=np.float32)
    merged = overlap_add_probabilities([left, right], [0, 4], [6, 6], 10)
    assert merged.shape == (10, 2)
    assert np.all(merged[:4] == 0.0)
    assert np.all(merged[6:] == 1.0)
    assert np.all((merged[4:6] > 0.0) & (merged[4:6] < 1.0))


def test_long_runtime_and_streaming_share_artifact(tmp_path: Path):
    deployment = SEDDeployment(_artifact(tmp_path / "model.pt"))
    samples = np.zeros(12_000, dtype=np.float32)
    result = deployment.predict_samples(
        samples,
        window_seconds=1.0,
        hop_seconds=0.5,
        include_unknown=False,
    )
    assert result["stats"]["windows"] == 2
    assert result["probabilities"].shape == (75, 2)

    processor = StreamingSEDProcessor(
        deployment,
        chunk_seconds=0.2,
        context_seconds=1.0,
        finalize_delay_seconds=0.1,
    )
    events = []
    for start in range(0, len(samples), 1600):
        events.extend(processor.process_chunk(samples[start : start + 1600]))
    events.extend(processor.flush())
    assert processor.processed_seconds == 1.5
    assert len(processor._history) <= 8000
    assert len({(row["label"], row["onset_s"], row["offset_s"]) for row in events}) == len(events)
