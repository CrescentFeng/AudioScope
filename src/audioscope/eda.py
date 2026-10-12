from __future__ import annotations

import json
from collections import Counter
from html import escape
from pathlib import Path
from typing import List

import numpy as np

from .audio import read_wav_mono, rms_dbfs
from .manifest import AudioRecord


def _write_svg(path: Path, class_counts: dict, loudness: List[float]) -> None:
    width, height = 920, 360
    margin = 48
    left_width = 420
    labels = list(class_counts)
    max_count = max(class_counts.values())
    bar_width = max(20, int((left_width - 2 * margin) / max(1, len(labels))))
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        "<style>text{font-family:system-ui,sans-serif;fill:#243238}.title{font-size:18px;font-weight:700}.label{font-size:12px}</style>",
        '<text class="title" x="48" y="30">Class balance</text>',
        '<text class="title" x="500" y="30">RMS loudness by clip</text>',
    ]
    for index, label in enumerate(labels):
        x = margin + index * bar_width
        bar_height = 240 * class_counts[label] / max_count
        y = 300 - bar_height
        elements.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bar_width - 8}" '
            f'height="{bar_height:.1f}" fill="#2a9d8f" rx="3"/>'
        )
        elements.append(f'<text class="label" x="{x}" y="322">{escape(label)}</text>')
        elements.append(f'<text class="label" x="{x}" y="{y - 6:.1f}">{class_counts[label]}</text>')
    low = min(loudness)
    high = max(loudness)
    spread = max(high - low, 1.0)
    for index, value in enumerate(loudness):
        x = 500 + 360 * index / max(1, len(loudness) - 1)
        y = 300 - 240 * (value - low) / spread
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#e76f51"/>')
    elements.append(f'<text class="label" x="500" y="322">min {low:.1f} dBFS</text>')
    elements.append(f'<text class="label" x="760" y="322">max {high:.1f} dBFS</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def build_eda(records: List[AudioRecord], data_root: Path, report_dir: Path) -> dict:
    durations, loudness, peaks = [], [], []
    anomalies = []
    for record in records:
        samples, _ = read_wav_mono(data_root / record.path, record.sample_rate)
        measured_duration = len(samples) / record.sample_rate
        peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
        durations.append(measured_duration)
        loudness.append(rms_dbfs(samples))
        peaks.append(peak)
        if abs(measured_duration - record.duration_s) > 0.02:
            anomalies.append({"asset_id": record.asset_id, "reason": "duration-mismatch"})
        if peak >= 0.999:
            anomalies.append({"asset_id": record.asset_id, "reason": "possible-clipping"})
        if rms_dbfs(samples) < -60.0:
            anomalies.append({"asset_id": record.asset_id, "reason": "near-silence"})
    result = {
        "records": len(records),
        "class_counts": dict(sorted(Counter(r.label for r in records).items())),
        "fold_counts": {
            str(key): value for key, value in sorted(Counter(r.fold for r in records).items())
        },
        "duration_s": {
            "min": min(durations),
            "mean": float(np.mean(durations)),
            "max": max(durations),
        },
        "rms_dbfs": {"min": min(loudness), "mean": float(np.mean(loudness)), "max": max(loudness)},
        "peak": {"max": max(peaks)},
        "anomalies": anomalies,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "eda.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_svg(report_dir / "eda.svg", result["class_counts"], loudness)
    return result
