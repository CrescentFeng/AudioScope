import numpy as np
import pytest

from audioscope.sed_metrics import probabilities_to_events
from audioscope.software_validation import CONDITIONS, transform
from audioscope.stream_decoder import IncrementalDecoder


def test_ten_thousand_processor_blocks_bound_all_state(monkeypatch):
    from audioscope.engineering import StreamingSEDProcessor

    class FakeDeployment:
        sample_rate = 50
        class_names = ["continuous"]
        feature_settings = {"hop_ms": 20}
        artifact = {"threshold": 0.5, "postprocess_settings": {"smooth_frames": 3}}

        def predict_feature(self, feature):
            return np.ones((500, 1), dtype=np.float32)

    monkeypatch.setattr("audioscope.engineering.extract_logmel", lambda *args: None)
    stream = StreamingSEDProcessor(FakeDeployment(), chunk_seconds=0.02)
    for _ in range(10000):
        assert stream.process_chunk(np.ones(1, dtype=np.float32)) == []
    assert stream.state_sizes() == {
        "audio_samples": 500,
        "pending_samples": 0,
        "median_frames": 3,
        "class_states": 1,
        "latency_entries": 2048,
    }
    assert stream.flush() == [{"label": "continuous", "onset_s": 0.0, "offset_s": 200.0}]
    assert stream.flush() == []


@pytest.mark.parametrize("condition", CONDITIONS)
def test_domain_shift_reproducibility(condition):
    samples = np.random.default_rng(42).normal(0, 0.1, 16000).astype(np.float32)
    first = transform(samples, 16000, condition)
    assert np.array_equal(first, transform(samples, 16000, condition))
    assert first.shape == samples.shape
    assert np.isfinite(first).all()
    assert np.abs(first).max() < 1


@pytest.mark.parametrize("smooth", [1, 3, 5, 7])
@pytest.mark.parametrize("frames", [1, 2, 1001])
def test_incremental_matches_batch(smooth, frames):
    probabilities = np.random.default_rng(41).random((frames, 2)).astype(np.float32)
    # Long active runs, small gaps, partial tail and simultaneous classes.
    probabilities[10:70, 0] = 0.9
    probabilities[35:38, 0] = 0.1
    probabilities[55:100, 1] = 0.9
    settings = {"smooth_frames": smooth, "merge_gap_s": 0.1, "minimum_duration_s": 0.06}
    threshold = {"one": 0.5, "two": 0.65}
    decoder = IncrementalDecoder(["one", "two"], threshold, 0.02, settings)
    events = []
    for i in range(0, frames, 13):
        events.extend(decoder.push(probabilities[i : i + 13]))
    duration = frames * 0.02 - 0.005
    events.extend(decoder.flush(duration))
    expected = probabilities_to_events(
        probabilities, ["one", "two"], threshold, 0.02, duration, **settings
    )
    actual = sorted((e["label"], e["onset_s"], e["offset_s"]) for e in events)
    assert actual == sorted((e.label, e.onset_s, e.offset_s) for e in expected)
    assert decoder.flush(duration) == []


def test_ten_thousand_blocks_keep_decoder_bounded():
    decoder = IncrementalDecoder(
        ["one", "two"], 0.5, 0.02, {"smooth_frames": 3, "merge_gap_s": 0.1}
    )
    block = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    for _ in range(10000):
        decoder.push(block)
        assert len(decoder.window) <= 3
        assert len(decoder.starts) == len(decoder.ends) == 2
    decoder.flush(400)
