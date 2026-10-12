from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .audio import read_wav_mono
from .features import extract_logmel
from .sed import build_sed_model
from .sed_data import TimedEvent
from .sed_metrics import probabilities_to_events
from .stream_decoder import IncrementalDecoder


def window_starts(total_samples: int, window_samples: int, hop_samples: int) -> list[int]:
    """Return deterministic, left-aligned window starts with complete tail coverage."""
    if total_samples < 0 or window_samples <= 0 or hop_samples <= 0:
        raise ValueError("sample counts must be non-negative and window/hop must be positive")
    if total_samples <= window_samples:
        return [0]
    return list(range(0, total_samples - window_samples + hop_samples, hop_samples))


def overlap_weights(frames: int, floor: float = 0.1) -> np.ndarray:
    """Triangular overlap-add weights that never mute the boundary frame."""
    if frames <= 0:
        raise ValueError("frames must be positive")
    if frames == 1:
        return np.ones(1, dtype=np.float32)
    centre = (frames - 1) / 2.0
    distance = np.abs(np.arange(frames, dtype=np.float32) - centre) / max(centre, 1.0)
    return np.maximum(1.0 - distance, float(floor)).astype(np.float32)


def overlap_add_probabilities(
    windows: Sequence[np.ndarray],
    start_frames: Sequence[int],
    valid_frames: Sequence[int],
    total_frames: int,
) -> np.ndarray:
    """Merge frame probabilities before event decoding to avoid seam duplicates."""
    if not (len(windows) == len(start_frames) == len(valid_frames)):
        raise ValueError("windows, start_frames and valid_frames must have equal length")
    if not windows:
        raise ValueError("at least one probability window is required")
    classes = windows[0].shape[1]
    weighted = np.zeros((total_frames, classes), dtype=np.float64)
    weights = np.zeros(total_frames, dtype=np.float64)
    for probabilities, start, valid in zip(windows, start_frames, valid_frames):
        if probabilities.ndim != 2 or probabilities.shape[1] != classes:
            raise ValueError("all windows must have shape [frames, classes]")
        usable = min(int(valid), len(probabilities), max(0, total_frames - int(start)))
        if usable <= 0:
            continue
        frame_weights = overlap_weights(usable)
        weighted[start : start + usable] += probabilities[:usable] * frame_weights[:, None]
        weights[start : start + usable] += frame_weights
    if np.any(weights == 0):
        raise ValueError("window layout left uncovered output frames")
    return (weighted / weights[:, None]).astype(np.float32)


def frame_rms_dbfs(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: int,
    hop_ms: int,
    total_frames: int,
) -> np.ndarray:
    frame_samples = max(1, round(sample_rate * frame_ms / 1000))
    hop_samples = max(1, round(sample_rate * hop_ms / 1000))
    values = np.empty(total_frames, dtype=np.float32)
    for index in range(total_frames):
        start = index * hop_samples
        frame = samples[start : start + frame_samples]
        if not len(frame):
            values[index] = -120.0
            continue
        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        values[index] = 20.0 * math.log10(max(rms, 1e-6))
    return values


def _threshold_for_profile(artifact: Mapping[str, Any], profile: str):
    profiles = artifact.get("threshold_profiles", {})
    if profile in profiles:
        return profiles[profile]
    if profile == "overall":
        return artifact["threshold"]
    raise ValueError(f"Threshold profile {profile!r} is unavailable")


def _event_dict(event: TimedEvent, confidence: float | None = None) -> dict:
    result = {
        "label": event.label,
        "onset_s": event.onset_s,
        "offset_s": event.offset_s,
    }
    if confidence is not None:
        result["confidence"] = round(float(confidence), 6)
    return result


@dataclass(frozen=True)
class InferenceStats:
    audio_seconds: float
    feature_seconds: float
    model_seconds: float
    total_seconds: float
    windows: int

    @property
    def rtf(self) -> float:
        return self.total_seconds / max(self.audio_seconds, 1e-9)


class SEDDeployment:
    """CPU deployment runtime shared by file, streaming, ONNX and demo paths."""

    def __init__(self, artifact_path: Path, onnx_path: Path | None = None) -> None:
        self.artifact_path = Path(artifact_path)
        self.artifact = torch.load(self.artifact_path, map_location="cpu", weights_only=True)
        self.class_names = list(self.artifact["class_names"])
        self.feature_settings = dict(self.artifact["feature_settings"])
        self.sample_rate = int(self.artifact["sample_rate"])
        self.mean, self.std = (float(value) for value in self.artifact["normalization"])
        calibration = self.artifact.get("calibration", {})
        self.temperature = float(calibration.get("temperature", 1.0))
        self.open_set_threshold = float(
            self.artifact.get("open_set", {}).get("known_confidence_threshold", 0.0)
        )
        self.activity_dbfs = float(self.artifact.get("open_set", {}).get("activity_dbfs", -50.0))
        self.onnx_path = Path(onnx_path) if onnx_path else None
        self.provenance = {
            "checkpoint_sha256": hashlib.sha256(self.artifact_path.read_bytes()).hexdigest(),
            "onnx_sha256": (
                hashlib.sha256(self.onnx_path.read_bytes()).hexdigest() if self.onnx_path else None
            ),
        }
        self.model = None
        self.session = None
        if self.onnx_path is None:
            model = build_sed_model(
                self.feature_settings,
                self.artifact["model_settings"],
                len(self.class_names),
            )
            model.load_state_dict(self.artifact["model_state"])
            model.eval()
            self.model = model
        else:
            try:
                import onnxruntime as ort
            except ImportError as error:  # pragma: no cover - exercised by optional install
                raise RuntimeError(
                    "Install AudioScope's engineering dependencies for ONNX"
                ) from error
            options = ort.SessionOptions()
            options.intra_op_num_threads = int(
                self.artifact.get("model_settings", {}).get("torch_threads", 8)
            )
            self.session = ort.InferenceSession(
                str(self.onnx_path), options, providers=["CPUExecutionProvider"]
            )

    @property
    def backend(self) -> str:
        return "onnxruntime" if self.session is not None else "pytorch"

    def predict_feature(self, feature: np.ndarray) -> np.ndarray:
        inputs = ((feature.astype(np.float32) - self.mean) / self.std)[None, None]
        if self.session is not None:
            logits = self.session.run(None, {self.session.get_inputs()[0].name: inputs})[0]
        else:
            assert self.model is not None
            with torch.inference_mode():
                logits = self.model(torch.from_numpy(inputs)).numpy()
        logits = logits[0].astype(np.float32) / self.temperature
        return (1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))).astype(np.float32)

    def predict_samples(
        self,
        samples: np.ndarray,
        threshold_profile: str = "overall",
        window_seconds: float = 10.0,
        hop_seconds: float = 5.0,
        include_unknown: bool = True,
    ) -> dict:
        started = time.perf_counter()
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        duration_s = len(samples) / self.sample_rate
        feature_hop_s = float(self.feature_settings["hop_ms"]) / 1000.0
        frame_hop_samples = round(self.sample_rate * feature_hop_s)
        target_frames = int(self.feature_settings["target_frames"])
        model_window_s = target_frames * feature_hop_s
        if abs(window_seconds - model_window_s) > feature_hop_s:
            raise ValueError(f"window_seconds must match the trained {model_window_s:.3f}s context")
        window_samples = round(window_seconds * self.sample_rate)
        hop_samples = round(hop_seconds * self.sample_rate)
        if hop_samples > window_samples:
            raise ValueError("hop_seconds must not exceed window_seconds")
        if hop_samples % frame_hop_samples:
            raise ValueError("hop_seconds must align with the feature hop")
        starts = window_starts(len(samples), window_samples, hop_samples)
        probability_windows: list[np.ndarray] = []
        valid_frames: list[int] = []
        start_frames: list[int] = []
        feature_seconds = 0.0
        model_seconds = 0.0
        for start in starts:
            clip = samples[start : start + window_samples]
            feature_started = time.perf_counter()
            feature = extract_logmel(clip, self.sample_rate, self.feature_settings)
            feature_seconds += time.perf_counter() - feature_started
            model_started = time.perf_counter()
            probability_windows.append(self.predict_feature(feature))
            model_seconds += time.perf_counter() - model_started
            valid_frames.append(
                min(target_frames, max(1, int(math.ceil(len(clip) / frame_hop_samples))))
            )
            start_frames.append(start // frame_hop_samples)
        total_frames = max(1, int(math.ceil(len(samples) / frame_hop_samples)))
        probabilities = overlap_add_probabilities(
            probability_windows, start_frames, valid_frames, total_frames
        )
        threshold = _threshold_for_profile(self.artifact, threshold_profile)
        postprocess = self.artifact["postprocess_settings"]
        events = probabilities_to_events(
            probabilities,
            self.class_names,
            threshold,
            feature_hop_s,
            duration_s,
            smooth_frames=int(postprocess.get("smooth_frames", 1)),
            merge_gap_s=float(postprocess.get("merge_gap_s", 0.2)),
            minimum_duration_s=float(postprocess.get("minimum_duration_s", 0.1)),
        )
        event_rows = []
        for event in events:
            class_index = self.class_names.index(event.label)
            left = max(0, int(event.onset_s / feature_hop_s))
            right = max(left + 1, int(math.ceil(event.offset_s / feature_hop_s)))
            row = _event_dict(event, probabilities[left:right, class_index].max())
            row["mean_confidence"] = round(
                float(probabilities[left:right, class_index].mean()), 6
            )
            event_rows.append(row)

        unknown_rows: list[dict] = []
        if include_unknown and self.open_set_threshold > 0:
            levels = frame_rms_dbfs(
                samples,
                self.sample_rate,
                int(self.feature_settings["frame_ms"]),
                int(self.feature_settings["hop_ms"]),
                total_frames,
            )
            unknown_score = 1.0 - probabilities.max(axis=1)
            unknown_score[levels < self.activity_dbfs] = 0.0
            unknown = probabilities_to_events(
                unknown_score[:, None],
                ["unknown"],
                1.0 - self.open_set_threshold,
                feature_hop_s,
                duration_s,
                smooth_frames=int(postprocess.get("smooth_frames", 1)),
                merge_gap_s=float(postprocess.get("merge_gap_s", 0.2)),
                minimum_duration_s=max(0.2, float(postprocess.get("minimum_duration_s", 0.1))),
            )
            unknown_rows = [_event_dict(event) for event in unknown]
        total_seconds = time.perf_counter() - started
        stats = InferenceStats(
            audio_seconds=duration_s,
            feature_seconds=feature_seconds,
            model_seconds=model_seconds,
            total_seconds=total_seconds,
            windows=len(starts),
        )
        return {
            "backend": self.backend,
            "threshold_profile": threshold_profile,
            "events": event_rows,
            "unknown_events": unknown_rows,
            "stats": {
                "audio_seconds": stats.audio_seconds,
                "windows": stats.windows,
                "feature_seconds": stats.feature_seconds,
                "model_seconds": stats.model_seconds,
                "total_seconds": stats.total_seconds,
                "rtf": stats.rtf,
            },
            "probabilities": probabilities,
        }

    def predict_file(self, audio_path: Path, **kwargs: Any) -> dict:
        samples, _ = read_wav_mono(Path(audio_path), self.sample_rate)
        result = self.predict_samples(samples, **kwargs)
        result["audio_path"] = str(Path(audio_path).resolve())
        return result


class StreamingSEDProcessor:
    """Bounded-audio-buffer streaming adapter with finalized-event emission."""

    def __init__(
        self,
        deployment: SEDDeployment,
        threshold_profile: str = "overall",
        chunk_seconds: float = 1.0,
        context_seconds: float = 10.0,
        finalize_delay_seconds: float = 0.4,
    ) -> None:
        self.deployment = deployment
        self.threshold_profile = threshold_profile
        self.chunk_samples = round(chunk_seconds * deployment.sample_rate)
        self.context_samples = round(context_seconds * deployment.sample_rate)
        self.finalize_delay_seconds = float(finalize_delay_seconds)
        if self.chunk_samples <= 0 or self.chunk_samples > self.context_samples:
            raise ValueError("stream chunk must be positive and no longer than context")
        feature_hop_samples = round(
            deployment.sample_rate * float(deployment.feature_settings["hop_ms"]) / 1000.0
        )
        if self.chunk_samples % feature_hop_samples:
            raise ValueError("chunk_seconds must align with the feature hop")
        self._feature_hop_samples = feature_hop_samples
        self._pending = np.empty(0, dtype=np.float32)
        self._history = np.empty(0, dtype=np.float32)
        self._processed_samples = 0
        self.inference_seconds = deque(maxlen=2048)
        self._closed = False
        self.decoder = IncrementalDecoder(
            deployment.class_names,
            _threshold_for_profile(deployment.artifact, threshold_profile),
            feature_hop_samples / deployment.sample_rate,
            deployment.artifact["postprocess_settings"],
            finalize_delay_seconds,
        )

    @property
    def processed_seconds(self) -> float:
        return self._processed_samples / self.deployment.sample_rate

    def _process_block(self, block: np.ndarray) -> list[dict]:
        context = np.concatenate([self._history, block])
        if len(context) > self.context_samples:
            context = context[-self.context_samples :]
        local_start = len(context) - len(block)
        started = time.perf_counter()
        feature = extract_logmel(
            context, self.deployment.sample_rate, self.deployment.feature_settings
        )
        probabilities = self.deployment.predict_feature(feature)
        self.inference_seconds.append(time.perf_counter() - started)
        start_frame = local_start // self._feature_hop_samples
        frames = max(1, int(math.ceil(len(block) / self._feature_hop_samples)))
        self._processed_samples += len(block)
        self._history = context[-self.context_samples :].copy()
        return self.decoder.push(probabilities[start_frame : start_frame + frames])

    def process_chunk(self, samples: np.ndarray) -> list[dict]:
        if self._closed:
            raise ValueError("stream has been flushed; create a new processor")
        chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
        rows = []
        for start in range(0, len(chunk), self.chunk_samples):
            combined = np.concatenate([self._pending, chunk[start : start + self.chunk_samples]])
            if len(combined) >= self.chunk_samples:
                rows.extend(self._process_block(combined[: self.chunk_samples]))
                self._pending = combined[self.chunk_samples :].copy()
            else:
                self._pending = combined
        return rows

    def flush(self) -> list[dict]:
        if self._closed:
            return []
        rows = []
        if len(self._pending):
            rows.extend(self._process_block(self._pending))
            self._pending = np.empty(0, dtype=np.float32)
        rows.extend(self.decoder.flush(self.processed_seconds))
        self._closed = True
        return rows

    def state_sizes(self) -> dict:
        return {
            "audio_samples": len(self._history),
            "pending_samples": len(self._pending),
            "median_frames": len(self.decoder.window),
            "class_states": len(self.decoder.starts),
            "latency_entries": len(self.inference_seconds),
        }


def stream_file(
    audio_path: Path,
    artifact_path: Path,
    threshold_profile: str = "overall",
    chunk_seconds: float = 1.0,
    onnx_path: Path | None = None,
) -> dict:
    deployment = SEDDeployment(artifact_path, onnx_path)
    samples, _ = read_wav_mono(audio_path, deployment.sample_rate)
    chunk_samples = round(chunk_seconds * deployment.sample_rate)
    processor = StreamingSEDProcessor(
        deployment, threshold_profile=threshold_profile, chunk_seconds=chunk_seconds
    )
    events = []
    for start in range(0, len(samples), chunk_samples):
        events.extend(processor.process_chunk(samples[start : start + chunk_samples]))
    events.extend(processor.flush())
    latency_ms = np.asarray(processor.inference_seconds, dtype=np.float64) * 1000.0
    return {
        "audio_path": str(Path(audio_path).resolve()),
        "backend": deployment.backend,
        "chunk_seconds": chunk_seconds,
        "algorithmic_latency_ms": chunk_seconds * 1000.0,
        "compute_latency_ms": {
            "median": float(np.median(latency_ms)) if len(latency_ms) else 0.0,
            "p95": float(np.percentile(latency_ms, 95)) if len(latency_ms) else 0.0,
        },
        "processed_seconds": processor.processed_seconds,
        "events": sorted(events, key=lambda item: (item["onset_s"], item["label"])),
    }


def stream_microphone(
    artifact_path: Path,
    duration_seconds: float,
    threshold_profile: str = "overall",
    chunk_seconds: float = 1.0,
    onnx_path: Path | None = None,
) -> dict:
    """Capture a bounded microphone session and emit finalized events as chunks arrive."""
    try:
        import sounddevice as sd
    except ImportError as error:  # pragma: no cover - optional hardware dependency
        raise RuntimeError(
            "Install AudioScope's microphone extra: pip install sounddevice"
        ) from error
    deployment = SEDDeployment(artifact_path, onnx_path)
    processor = StreamingSEDProcessor(
        deployment, threshold_profile=threshold_profile, chunk_seconds=chunk_seconds
    )
    block_samples = round(chunk_seconds * deployment.sample_rate)
    blocks = max(1, int(math.ceil(duration_seconds / chunk_seconds)))
    events = []
    started = time.perf_counter()
    with sd.InputStream(
        samplerate=deployment.sample_rate,
        channels=1,
        dtype="float32",
        blocksize=block_samples,
    ) as stream:
        for _ in range(blocks):
            samples, overflowed = stream.read(block_samples)
            if overflowed:
                raise RuntimeError("microphone input overflowed; increase chunk_seconds")
            events.extend(processor.process_chunk(samples[:, 0]))
    events.extend(processor.flush())
    return {
        "backend": deployment.backend,
        "recorded_seconds": processor.processed_seconds,
        "wall_seconds": time.perf_counter() - started,
        "chunk_seconds": chunk_seconds,
        "events": sorted(events, key=lambda item: (item["onset_s"], item["label"])),
    }
