import numpy as np

from audioscope.field_ops import _balanced_test_selection, estimate_sync_offset


def test_sync_offset_recovers_delayed_reference():
    sample_rate = 1000
    rng = np.random.default_rng(2027)
    reference = rng.normal(0, 0.2, sample_rate * 2).astype(np.float32)
    recording = np.concatenate([np.zeros(375, dtype=np.float32), reference, np.zeros(500)])
    assert abs(estimate_sync_offset(recording, reference, sample_rate) - 0.375) < 0.002


def test_balanced_selection_is_deterministic_and_covers_classes():
    records = [
        {
            "scene_id": f"scene-{index:02d}",
            "split": "test",
            "events": [{"label": f"class-{index % 4}"}],
        }
        for index in range(12)
    ]
    first = _balanced_test_selection(records, 4)
    second = _balanced_test_selection(records, 4)
    assert first == second
    assert {row["events"][0]["label"] for row in first} == {
        "class-0",
        "class-1",
        "class-2",
        "class-3",
    }
