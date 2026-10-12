"""Reference comparison and interpretable error triage, never used as training labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .audio import read_wav_mono
from .sed_data import SoundscapeRecord, TimedEvent, read_soundscape_manifest
from .sed_metrics import evaluate_sed
from .workbench import LABELS_ZH, sha256, validate_events


def overlap(a, b):
    return max(0.0, min(a["offset_s"], b["offset_s"]) - max(a["onset_s"], b["onset_s"]))


def error_triage(reference: list[dict], prediction: list[dict]) -> dict:
    """Maximum-cardinality same-class collar matching, then mutually exclusive triage.

    Triage categories are heuristics, not proof of the physical cause of an error.
    """
    prediction = [e for e in prediction if e["label"] != "unknown"]
    matched_r, matched_p, matches = set(), set(), []
    if reference and prediction:
        cost = np.full((len(reference), len(prediction)), 1e9, dtype=np.float64)
        for i, ref in enumerate(reference):
            for j, pred in enumerate(prediction):
                onset_error = abs(ref["onset_s"] - pred["onset_s"])
                offset_error = abs(ref["offset_s"] - pred["offset_s"])
                collar = max(0.2, 0.2 * (ref["offset_s"] - ref["onset_s"]))
                if ref["label"] == pred["label"] and onset_error <= 0.2 and offset_error <= collar:
                    cost[i, j] = onset_error + offset_error
        rows, columns = linear_sum_assignment(cost)
        for i, j in zip(rows.tolist(), columns.tolist()):
            if cost[i, j] < 1e9:
                matched_r.add(i)
                matched_p.add(j)
                matches.append(
                    {
                        "reference_index": i,
                        "prediction_index": j,
                        "onset_error_s": prediction[j]["onset_s"] - reference[i]["onset_s"],
                        "offset_error_s": prediction[j]["offset_s"] - reference[i]["offset_s"],
                    }
                )
    missed, extra = [], []
    for i, ref in enumerate(reference):
        if i in matched_r:
            continue
        same = [p for p in prediction if p["label"] == ref["label"] and overlap(ref, p) > 0]
        category = (
            "fragmented_or_boundary"
            if len(same) > 1
            else "boundary"
            if same
            else "possible_class_confusion"
            if any(overlap(ref, p) > 0 for p in prediction)
            else "missed"
        )
        missed.append({"reference_index": i, "category": category, **ref})
    for j, pred in enumerate(prediction):
        if j in matched_p:
            continue
        same = [r for r in reference if r["label"] == pred["label"] and overlap(r, pred) > 0]
        category = (
            "duplicate_or_boundary"
            if same
            else "possible_class_confusion"
            if any(overlap(r, pred) > 0 for r in reference)
            else "spurious"
        )
        extra.append({"prediction_index": j, "category": category, **pred})
    return {
        "tp": len(matches),
        "fn": len(missed),
        "fp": len(extra),
        "matched": matches,
        "missed_reference": missed,
        "extra_prediction": extra,
        "note": "错因分类为重叠关系启发式；FN 与 FP 可来自同一个边界错误，不应相加当事件数。",
    }


def compare_reference(job: dict, document: dict) -> dict:
    payload = job["payload"]
    if document.get("audio_sha256") != payload["audio_sha256"]:
        raise ValueError("参考标注必须包含与当前音频一致的 audio_sha256。")
    if document.get("source") not in {"dataset_reference", "human_reference"}:
        raise ValueError(
            "参考来源只能是 dataset_reference 或 human_reference，不接受模型预测冒充真值。"
        )
    if document.get("complete") is not True:
        raise ValueError("参考标注必须声明 complete=true；不完整标注不能用于计算准确性指标。")
    reference = validate_events(document["events"], payload["duration_s"])
    if any(e["label"] == "unknown" for e in reference):
        raise ValueError("当前参考指标只评估十个已知类别，参考事件不能使用 unknown。")

    def timed(events):
        return [
            TimedEvent(e["label"], e["onset_s"], e["offset_s"], "reference", "reference")
            for e in events
            if e["label"] != "unknown"
        ]

    record = SoundscapeRecord(
        "review", "", "reference", payload["duration_s"], 16000, tuple(timed(reference))
    )
    metrics = evaluate_sed(
        [record],
        {"review": timed(payload["events"])},
        [k for k in LABELS_ZH if k != "unknown"],
        {"onset_collar_s": 0.2, "offset_collar_s": 0.2, "offset_ratio": 0.2},
        1.0,
    )
    return {
        "source": document["source"],
        "metrics": metrics,
        "reference_events": reference,
        "triage": error_triage(reference, payload["events"]),
        "notice": "这是当前音频的参考对照，不是整个模型准确率；不用于训练或选择阈值。",
    }


def load_reference(path: Path, job: dict) -> dict:
    if path.stat().st_size > 5 * 1024 * 1024:
        raise ValueError("参考 JSON 不能超过 5 MiB。")
    try:
        return compare_reference(job, json.loads(path.read_text(encoding="utf-8")))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("参考 JSON 格式不正确。") from exc


def demo_reference(root: Path, job: dict) -> dict:
    """Reconstruct the existing 12-clip fixture, and prove waveform identity before use."""
    demo = root / "data/l4-engineering/l4_long_demo.wav"
    if job["payload"]["audio_sha256"] != sha256(demo):
        raise ValueError("内置参考只适用于项目原始示例音频；其他文件请上传自己的参考 JSON。")
    manifest = root / "data/desed-real-strong-only/manifest.jsonl"
    records = sorted(
        [r for r in read_soundscape_manifest(manifest) if r.split == "test"],
        key=lambda r: r.scene_id,
    )[:12]
    parts, events, offset = [], [], 0.0
    for record in records:
        samples, _ = read_wav_mono(manifest.parent / record.path, 16000)
        parts.append(samples)
        events.extend(
            {"label": e.label, "onset_s": e.onset_s + offset, "offset_s": e.offset_s + offset}
            for e in record.events
        )
        offset += len(samples) / 16000
    actual, _ = read_wav_mono(demo, 16000)
    if not np.array_equal(actual, np.concatenate(parts)):
        raise ValueError("示例音频不再与构建协议匹配，不能使用内置参考。")
    return compare_reference(
        job,
        {
            "audio_sha256": sha256(demo),
            "source": "dataset_reference",
            "complete": True,
            "events": events,
        },
    )


def compare_jobs(left: dict, right: dict) -> dict:
    if left["payload"]["audio_sha256"] != right["payload"]["audio_sha256"]:
        raise ValueError("只支持对同一个原始音频的不同分析任务进行比较。")
    if left["status"] != "completed" or right["status"] != "completed":
        raise ValueError("只能比较已完成的分析任务。")
    a = {(e["label"], e["onset_s"], e["offset_s"]) for e in left["payload"]["events"]}
    b = {(e["label"], e["onset_s"], e["offset_s"]) for e in right["payload"]["events"]}
    return {
        "left_id": left["id"],
        "right_id": right["id"],
        "left_profile": left["payload"]["threshold_profile"],
        "right_profile": right["payload"]["threshold_profile"],
        "left_model": left["payload"]["model"],
        "right_model": right["payload"]["model"],
        "exact_same_events": len(a & b),
        "left_only_events": len(a - b),
        "right_only_events": len(b - a),
        "notice": "这是模型/工作点差异，不代表谁更准确；精度必须另与完整参考标注对照。",
    }
