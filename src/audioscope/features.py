from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.fft import dct


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    mel_points = np.linspace(
        _hz_to_mel(np.array([0.0]))[0], _hz_to_mel(np.array([sample_rate / 2]))[0], n_mels + 2
    )
    bins = np.floor((n_fft + 1) * _mel_to_hz(mel_points) / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(1, n_mels + 1):
        left, center, right = bins[index - 1 : index + 2]
        if center > left:
            filters[index - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            filters[index - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return filters


def extract_mfcc_summary(
    samples: np.ndarray, sample_rate: int, settings: Mapping[str, int]
) -> np.ndarray:
    n_fft = int(settings["n_fft"])
    n_mels = int(settings["n_mels"])
    n_mfcc = int(settings["n_mfcc"])
    frame_length = max(1, round(sample_rate * int(settings["frame_ms"]) / 1000))
    hop_length = max(1, round(sample_rate * int(settings["hop_ms"]) / 1000))
    if len(samples) < frame_length:
        samples = np.pad(samples, (0, frame_length - len(samples)))
    frame_count = 1 + (len(samples) - frame_length) // hop_length
    shape = (frame_count, frame_length)
    strides = (samples.strides[0] * hop_length, samples.strides[0])
    frames = np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides).copy()
    frames *= np.hanning(frame_length).astype(np.float32)
    power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    mel_energy = power @ _mel_filterbank(sample_rate, n_fft, n_mels).T
    log_mel = np.log(np.maximum(mel_energy, 1e-10))
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0)]).astype(np.float32)


def extract_logmel(
    samples: np.ndarray,
    sample_rate: int,
    settings: Mapping[str, int],
) -> np.ndarray:
    """Return a fixed-size log-mel image shaped ``[n_mels, target_frames]``."""
    n_fft = int(settings["n_fft"])
    n_mels = int(settings["n_mels"])
    target_frames = int(settings["target_frames"])
    frame_length = max(1, round(sample_rate * int(settings["frame_ms"]) / 1000))
    hop_length = max(1, round(sample_rate * int(settings["hop_ms"]) / 1000))
    if n_fft < frame_length:
        raise ValueError("n_fft must be greater than or equal to the frame length")
    if len(samples) < frame_length:
        samples = np.pad(samples, (0, frame_length - len(samples)))
    frame_count = 1 + (len(samples) - frame_length) // hop_length
    shape = (frame_count, frame_length)
    strides = (samples.strides[0] * hop_length, samples.strides[0])
    frames = np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides).copy()
    frames *= np.hanning(frame_length).astype(np.float32)
    power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    mel_energy = power @ _mel_filterbank(sample_rate, n_fft, n_mels).T
    log_mel = np.log(np.maximum(mel_energy, 1e-10)).T.astype(np.float32)
    if log_mel.shape[1] < target_frames:
        pad_width = target_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="edge")
    elif log_mel.shape[1] > target_frames:
        start = (log_mel.shape[1] - target_frames) // 2
        log_mel = log_mel[:, start : start + target_frames]
    return log_mel
