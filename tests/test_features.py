import numpy as np

from audioscope.features import extract_logmel, extract_mfcc_summary


def test_feature_shape_and_finiteness():
    sample_rate = 16000
    time = np.arange(sample_rate) / sample_rate
    samples = np.sin(2 * np.pi * 440 * time).astype(np.float32)
    settings = {"n_mfcc": 20, "n_mels": 40, "n_fft": 512, "frame_ms": 25, "hop_ms": 10}
    features = extract_mfcc_summary(samples, sample_rate, settings)
    assert features.shape == (40,)
    assert np.isfinite(features).all()


def test_logmel_shape_and_finiteness():
    sample_rate = 8000
    time = np.arange(sample_rate // 4) / sample_rate
    samples = np.sin(2 * np.pi * 440 * time).astype(np.float32)
    settings = {
        "n_mels": 24,
        "n_fft": 256,
        "frame_ms": 25,
        "hop_ms": 10,
        "target_frames": 24,
    }
    features = extract_logmel(samples, sample_rate, settings)
    assert features.shape == (24, 24)
    assert np.isfinite(features).all()
