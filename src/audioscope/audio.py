from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def _to_float32(samples: np.ndarray) -> np.ndarray:
    if np.issubdtype(samples.dtype, np.floating):
        return np.clip(samples.astype(np.float32), -1.0, 1.0)
    info = np.iinfo(samples.dtype)
    scale = float(max(abs(info.min), info.max))
    return samples.astype(np.float32) / scale


def read_wav_mono(path: Path, target_sample_rate: int) -> Tuple[np.ndarray, int]:
    sample_rate, samples = wavfile.read(path)
    samples = _to_float32(np.asarray(samples))
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if sample_rate != target_sample_rate:
        factor = math.gcd(sample_rate, target_sample_rate)
        samples = resample_poly(samples, target_sample_rate // factor, sample_rate // factor)
    return np.asarray(samples, dtype=np.float32), target_sample_rate


def rms_dbfs(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    return 20.0 * math.log10(max(rms, 1e-12))


def add_white_noise(samples: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    signal_power = float(np.mean(np.square(samples), dtype=np.float64))
    if signal_power <= 1e-16:
        return samples.copy()
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, math.sqrt(noise_power), size=samples.shape)
    return np.clip(samples + noise, -1.0, 1.0).astype(np.float32)
