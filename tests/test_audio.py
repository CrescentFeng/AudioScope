import numpy as np

from audioscope.audio import add_white_noise, rms_dbfs


def test_noise_is_deterministic_and_changes_signal():
    samples = np.full(1600, 0.25, dtype=np.float32)
    first = add_white_noise(samples, 10.0, seed=7)
    second = add_white_noise(samples, 10.0, seed=7)
    np.testing.assert_allclose(first, second)
    assert not np.allclose(first, samples)


def test_rms_dbfs_for_half_scale_constant():
    assert abs(rms_dbfs(np.full(100, 0.5, dtype=np.float32)) + 6.0206) < 0.001
