from __future__ import annotations

import numpy as np
import pytest

from audioscope.sed_data import SoundscapeRecord, TimedEvent
from audioscope.sed_metrics import (
    duration_binned_event_metrics,
    evaluate_sed,
    probabilities_to_events,
)


def _event(label: str, onset: float, offset: float) -> TimedEvent:
    return TimedEvent(label, onset, offset, "source", "recording")


def test_postprocess_merges_short_gap():
    probabilities = np.zeros((20, 1), dtype=np.float32)
    probabilities[2:7, 0] = 0.9
    probabilities[8:12, 0] = 0.9
    events = probabilities_to_events(
        probabilities,
        ["dog"],
        threshold=0.5,
        hop_s=0.1,
        duration_s=2.0,
        merge_gap_s=0.15,
        minimum_duration_s=0.1,
    )
    assert len(events) == 1
    assert events[0].onset_s == 0.2
    assert events[0].offset_s == 1.2


def test_event_and_segment_metrics_with_overlap():
    record = SoundscapeRecord(
        scene_id="scene",
        path="scene.wav",
        split="test",
        duration_s=4.0,
        sample_rate=16000,
        events=(_event("dog", 0.5, 2.0), _event("rain", 1.0, 3.0)),
    )
    estimates = {
        "scene": [_event("dog", 0.6, 2.1), _event("rain", 1.1, 2.9)]
    }
    metrics = evaluate_sed(
        [record],
        estimates,
        ["dog", "rain"],
        {"onset_collar_s": 0.2, "offset_collar_s": 0.2, "offset_ratio": 0.2},
        1.0,
    )
    assert metrics["event_based"]["micro"]["f1"] == 1.0
    assert metrics["segment_based"]["micro"]["f1"] == pytest.approx(8 / 9)


def test_event_metric_rejects_outside_collar():
    record = SoundscapeRecord(
        "scene",
        "scene.wav",
        "test",
        3.0,
        16000,
        (_event("dog", 0.5, 1.5),),
    )
    metrics = evaluate_sed(
        [record],
        {"scene": [_event("dog", 0.8, 1.5)]},
        ["dog"],
        {"onset_collar_s": 0.2, "offset_collar_s": 0.2, "offset_ratio": 0.2},
        1.0,
    )
    assert metrics["event_based"]["micro"]["f1"] == 0.0


def test_per_class_thresholds_and_duration_limit():
    probabilities = np.zeros((20, 2), dtype=np.float32)
    probabilities[2:5, 0] = 0.7
    probabilities[8:12, 1] = 0.4
    probabilities[15:, 0] = 1.0  # Must be ignored because the clip ends at 1.5 s.
    events = probabilities_to_events(
        probabilities,
        ["dog", "rain"],
        threshold={"dog": 0.6, "rain": 0.3},
        hop_s=0.1,
        duration_s=1.5,
        merge_gap_s=0.0,
        minimum_duration_s=0.1,
    )
    assert [(event.label, event.onset_s, event.offset_s) for event in events] == [
        ("dog", 0.2, 0.5),
        ("rain", 0.8, 1.2),
    ]


def test_duration_binned_metrics_separate_short_and_long_events():
    record = SoundscapeRecord(
        "scene",
        "scene.wav",
        "test",
        4.0,
        16000,
        (_event("dog", 0.0, 0.5), _event("dog", 1.0, 3.0)),
    )
    metrics = duration_binned_event_metrics(
        [record],
        {"scene": [_event("dog", 0.0, 0.5)]},
        ["dog"],
        {"onset_collar_s": 0.2, "offset_collar_s": 0.2, "offset_ratio": 0.2},
        short_event_max_s=1.0,
    )
    assert metrics["short"]["micro"]["f1"] == 1.0
    assert metrics["long"]["micro"]["f1"] == 0.0
