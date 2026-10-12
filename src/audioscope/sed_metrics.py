from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment

from .sed_data import SoundscapeRecord, TimedEvent


def _runs(activity: np.ndarray) -> List[Tuple[int, int]]:
    padded = np.pad(activity.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _merge_runs(runs: Sequence[Tuple[int, int]], gap_frames: int) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= gap_frames:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def probabilities_to_events(
    probabilities: np.ndarray,
    class_names: Sequence[str],
    threshold: float | Mapping[str, float] | Sequence[float],
    hop_s: float,
    duration_s: float,
    smooth_frames: int = 1,
    merge_gap_s: float = 0.2,
    minimum_duration_s: float = 0.1,
) -> List[TimedEvent]:
    if probabilities.ndim != 2 or probabilities.shape[1] != len(class_names):
        raise ValueError("probabilities must have shape [frames, classes]")
    frame_limit = min(len(probabilities), int(math.ceil(duration_s / hop_s)))
    scores = probabilities[:frame_limit]
    if smooth_frames > 1:
        if smooth_frames % 2 == 0:
            raise ValueError("smooth_frames must be odd")
        scores = median_filter(scores, size=(smooth_frames, 1), mode="nearest")
    if isinstance(threshold, Mapping):
        threshold_values = np.asarray(
            [float(threshold[label]) for label in class_names], dtype=np.float32
        )
    elif np.isscalar(threshold):
        threshold_values = np.full(len(class_names), float(threshold), dtype=np.float32)
    else:
        threshold_values = np.asarray(threshold, dtype=np.float32)
        if threshold_values.shape != (len(class_names),):
            raise ValueError("threshold sequence must have one value per class")
    activity = scores >= threshold_values[None, :]
    gap_frames = int(round(merge_gap_s / hop_s))
    minimum_frames = max(1, int(math.ceil(minimum_duration_s / hop_s)))
    events = []
    for class_index, label in enumerate(class_names):
        for start, end in _merge_runs(_runs(activity[:, class_index]), gap_frames):
            if end - start < minimum_frames:
                continue
            events.append(
                TimedEvent(
                    label=label,
                    onset_s=round(start * hop_s, 6),
                    offset_s=round(min(duration_s, end * hop_s), 6),
                    source_asset_id="prediction",
                    source_recording_id="prediction",
                )
            )
    return sorted(events, key=lambda event: (event.onset_s, event.label))


def _match_events(
    reference: Sequence[TimedEvent],
    estimated: Sequence[TimedEvent],
    onset_collar_s: float,
    offset_collar_s: float,
    offset_ratio: float,
) -> int:
    if not reference or not estimated:
        return 0
    valid = np.zeros((len(reference), len(estimated)), dtype=bool)
    cost = np.full(valid.shape, 1_000_000.0, dtype=np.float64)
    for ref_index, ref in enumerate(reference):
        duration = ref.offset_s - ref.onset_s
        offset_tolerance = max(offset_collar_s, offset_ratio * duration)
        for est_index, est in enumerate(estimated):
            onset_error = abs(est.onset_s - ref.onset_s)
            offset_error = abs(est.offset_s - ref.offset_s)
            if onset_error <= onset_collar_s and offset_error <= offset_tolerance:
                valid[ref_index, est_index] = True
                cost[ref_index, est_index] = onset_error + offset_error
    rows, columns = linear_sum_assignment(cost)
    return sum(bool(valid[row, column]) for row, column in zip(rows, columns))


def _f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def event_based_metrics(
    references: Mapping[str, Sequence[TimedEvent]],
    estimates: Mapping[str, Sequence[TimedEvent]],
    class_names: Sequence[str],
    onset_collar_s: float = 0.2,
    offset_collar_s: float = 0.2,
    offset_ratio: float = 0.2,
) -> dict:
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in class_names}
    for scene_id in references:
        reference_events = references[scene_id]
        estimated_events = estimates.get(scene_id, ())
        for label in class_names:
            ref = [event for event in reference_events if event.label == label]
            est = [event for event in estimated_events if event.label == label]
            matches = _match_events(
                ref, est, onset_collar_s, offset_collar_s, offset_ratio
            )
            counts[label]["tp"] += matches
            counts[label]["fp"] += len(est) - matches
            counts[label]["fn"] += len(ref) - matches
    per_class = {
        label: _f1(values["tp"], values["fp"], values["fn"])
        for label, values in counts.items()
    }
    total = {
        key: sum(values[key] for values in counts.values()) for key in ("tp", "fp", "fn")
    }
    return {
        "micro": _f1(total["tp"], total["fp"], total["fn"]),
        "macro_f1": float(np.mean([values["f1"] for values in per_class.values()])),
        "per_class": per_class,
        "onset_collar_s": onset_collar_s,
        "offset_collar_s": offset_collar_s,
        "offset_ratio": offset_ratio,
    }


def _segment_activity(
    events: Iterable[TimedEvent],
    class_names: Sequence[str],
    duration_s: float,
    resolution_s: float,
) -> np.ndarray:
    segments = int(math.ceil(duration_s / resolution_s))
    activity = np.zeros((segments, len(class_names)), dtype=bool)
    class_to_index = {label: index for index, label in enumerate(class_names)}
    for event in events:
        start = max(0, int(math.floor(event.onset_s / resolution_s)))
        end = min(segments, int(math.ceil(event.offset_s / resolution_s)))
        activity[start:end, class_to_index[event.label]] = True
    return activity


def segment_based_metrics(
    records: Sequence[SoundscapeRecord],
    estimates: Mapping[str, Sequence[TimedEvent]],
    class_names: Sequence[str],
    resolution_s: float = 1.0,
) -> dict:
    class_counts = {
        label: {"tp": 0, "fp": 0, "fn": 0} for label in class_names
    }
    for record in records:
        reference = _segment_activity(
            record.events, class_names, record.duration_s, resolution_s
        )
        estimated = _segment_activity(
            estimates.get(record.scene_id, ()), class_names, record.duration_s, resolution_s
        )
        for class_index, label in enumerate(class_names):
            class_counts[label]["tp"] += int(
                np.logical_and(reference[:, class_index], estimated[:, class_index]).sum()
            )
            class_counts[label]["fp"] += int(
                np.logical_and(~reference[:, class_index], estimated[:, class_index]).sum()
            )
            class_counts[label]["fn"] += int(
                np.logical_and(reference[:, class_index], ~estimated[:, class_index]).sum()
            )
    per_class = {
        label: _f1(values["tp"], values["fp"], values["fn"])
        for label, values in class_counts.items()
    }
    total = {
        key: sum(values[key] for values in class_counts.values())
        for key in ("tp", "fp", "fn")
    }
    return {
        "micro": _f1(total["tp"], total["fp"], total["fn"]),
        "macro_f1": float(np.mean([values["f1"] for values in per_class.values()])),
        "per_class": per_class,
        "resolution_s": resolution_s,
    }


def evaluate_sed(
    records: Sequence[SoundscapeRecord],
    estimates: Mapping[str, Sequence[TimedEvent]],
    class_names: Sequence[str],
    event_settings: Mapping[str, float],
    segment_resolution_s: float,
) -> dict:
    references = {record.scene_id: record.events for record in records}
    return {
        "event_based": event_based_metrics(
            references,
            estimates,
            class_names,
            onset_collar_s=float(event_settings["onset_collar_s"]),
            offset_collar_s=float(event_settings["offset_collar_s"]),
            offset_ratio=float(event_settings["offset_ratio"]),
        ),
        "segment_based": segment_based_metrics(
            records, estimates, class_names, float(segment_resolution_s)
        ),
    }


def duration_binned_event_metrics(
    records: Sequence[SoundscapeRecord],
    estimates: Mapping[str, Sequence[TimedEvent]],
    class_names: Sequence[str],
    event_settings: Mapping[str, float],
    short_event_max_s: float,
) -> dict:
    """Report event metrics for short and long events using the same duration bins.

    Both reference and estimated events are assigned by their own duration. This makes the
    diagnostic strict: a boundary error that moves an event across the duration boundary is not
    counted as a match in either bin.
    """

    references = {record.scene_id: record.events for record in records}

    def select(events: Sequence[TimedEvent], short: bool) -> List[TimedEvent]:
        return [
            event
            for event in events
            if ((event.offset_s - event.onset_s <= short_event_max_s) == short)
        ]

    result = {}
    for name, short in (("short", True), ("long", False)):
        reference_subset = {
            scene_id: select(events, short) for scene_id, events in references.items()
        }
        estimate_subset = {
            scene_id: select(estimates.get(scene_id, ()), short)
            for scene_id in references
        }
        metrics = event_based_metrics(
            reference_subset,
            estimate_subset,
            class_names,
            onset_collar_s=float(event_settings["onset_collar_s"]),
            offset_collar_s=float(event_settings["offset_collar_s"]),
            offset_ratio=float(event_settings["offset_ratio"]),
        )
        metrics["reference_events"] = sum(map(len, reference_subset.values()))
        metrics["estimated_events"] = sum(map(len, estimate_subset.values()))
        result[name] = metrics
    result["short_event_max_s"] = float(short_event_max_s)
    return result
