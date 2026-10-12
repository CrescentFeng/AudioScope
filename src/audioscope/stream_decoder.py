"""Bounded, batch-equivalent median smoothing and event finalization."""

from __future__ import annotations

import math
from collections import deque

import numpy as np


class IncrementalDecoder:
    def __init__(self, labels, thresholds, hop_s, settings, delay_s=0.4):
        self.labels = list(labels)
        self.thresholds = np.asarray(
            [thresholds[label] for label in labels]
            if isinstance(thresholds, dict)
            else np.broadcast_to(thresholds, (len(labels),)),
            dtype=np.float32,
        )
        self.hop_s = hop_s
        self.width = int(settings.get("smooth_frames", 1))
        if self.width < 1 or self.width % 2 != 1:
            raise ValueError("smooth_frames must be positive and odd")
        self.radius = self.width // 2
        self.window = deque(maxlen=self.width)
        self.gap = round(settings.get("merge_gap_s", 0.2) / hop_s)
        self.wait = max(self.gap, math.ceil(delay_s / hop_s))
        self.minimum = max(1, math.ceil(settings.get("minimum_duration_s", 0.1) / hop_s))
        self.starts = [None] * len(labels)
        self.ends = [None] * len(labels)
        self.frames = 0
        self.input_frames = 0
        self.closed = False

    def _close(self, cls, duration_s=None):
        start, end = self.starts[cls], self.ends[cls]
        self.starts[cls] = self.ends[cls] = None
        if start is None or end - start < self.minimum:
            return []
        end_s = end * self.hop_s
        if duration_s is not None:
            end_s = min(end_s, duration_s)
        return [
            {
                "label": self.labels[cls],
                "onset_s": round(start * self.hop_s, 6),
                "offset_s": round(end_s, 6),
            }
        ]

    def _step(self):
        activity = np.median(np.asarray(self.window), axis=0) >= self.thresholds
        rows = []
        for cls, active in enumerate(activity):
            if active:
                if self.starts[cls] is not None and self.frames - self.ends[cls] > self.gap:
                    rows.extend(self._close(cls))
                if self.starts[cls] is None:
                    self.starts[cls] = self.frames
                self.ends[cls] = self.frames + 1
            elif self.starts[cls] is not None and self.frames + 1 - self.ends[cls] > self.wait:
                rows.extend(self._close(cls))
        self.frames += 1
        return rows

    def push(self, probabilities):
        if self.closed:
            raise ValueError("decoder has been flushed")
        rows = []
        for frame in probabilities:
            if self.input_frames == 0:
                self.window.extend(frame.copy() for _ in range(self.radius))
            self.window.append(frame.copy())
            self.input_frames += 1
            if len(self.window) == self.width:
                rows.extend(self._step())
        return rows

    def flush(self, duration_s):
        if self.closed:
            return []
        self.closed = True
        rows = []
        if self.input_frames:
            last = self.window[-1].copy()
            while self.frames < self.input_frames:
                self.window.append(last)
                if len(self.window) == self.width:
                    rows.extend(self._step())
            for cls in range(len(self.labels)):
                rows.extend(self._close(cls, duration_s))
        return sorted(rows, key=lambda event: (event["onset_s"], event["label"]))
