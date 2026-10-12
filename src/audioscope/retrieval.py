"""Explicit known-label queries and offline CLAP similarity retrieval (not classification)."""

from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np

from .audio import read_wav_mono
from .workbench import LABELS_ZH, sha256, validate_audio

ALIASES = {
    "alarm_bell_ringing": ("铃声", "闹钟", "alarm", "bell"),
    "blender": ("搅拌机", "blender"),
    "cat": ("猫", "cat"),
    "dog": ("狗", "dog"),
    "dishes": ("餐具", "碗碟", "dishes"),
    "electric_shaver_toothbrush": ("剃须刀", "牙刷", "shaver", "toothbrush"),
    "frying": ("煎", "炒", "frying"),
    "running_water": ("流水", "水声", "water"),
    "speech": ("说话", "人声", "speech", "talking"),
    "vacuum_cleaner": ("吸尘器", "vacuum"),
    "unknown": ("未知", "待确认", "unknown"),
}


def query_events(events: list[dict], query: str, require_overlap=False) -> list[dict]:
    text = query.strip().lower()
    if not text:
        raise ValueError("请输入检索内容。")
    if any(word in text for word in ("没有", "不含", "排除", "without", "not ")) or re.search(
        r"\bno\b", text
    ):
        raise ValueError("当前类别查询不支持否定条件；请使用明确的类别及‘同时出现’选项。")

    def contains(word):
        if word.isascii():
            return re.search(r"(?<![a-z])" + re.escape(word) + r"s?(?![a-z])", text) is not None
        return word in text

    labels = [label for label, words in ALIASES.items() if any(contains(w) for w in words)]
    if not labels:
        raise ValueError("未识别到已知类别。开放描述请使用 CLAP 检索，不会自动猜测类别。")
    require_overlap = require_overlap or any(w in text for w in ("同时", "伴随", "together"))
    selected = [dict(e, query_source="known_event_filter") for e in events if e["label"] in labels]
    if not require_overlap or len(labels) == 1:
        return sorted(selected, key=lambda e: (e["onset_s"], e["label"]))
    spans = [(e["onset_s"], e["offset_s"]) for e in selected if e["label"] == labels[0]]
    for label in labels[1:]:
        spans = [
            (max(a, e["onset_s"]), min(b, e["offset_s"]))
            for a, b in spans
            for e in selected
            if e["label"] == label and max(a, e["onset_s"]) < min(b, e["offset_s"])
        ]
    return [
        {
            "label": " + ".join(labels),
            "onset_s": a,
            "offset_s": b,
            "query_source": "predicted_overlap",
            "confidence": None,
        }
        for a, b in sorted(set(spans))
    ]


class ClapSearch:
    """One in-memory audio index; no download, no disk embedding cache or raw retention."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = self.processor = None
        self.lock = threading.Lock()
        self.index_key = None
        self.embeddings = None
        self.spans = []
        self.provenance = None

    def _load(self):
        if self.model is None:
            if not (self.model_path / "config.json").is_file():
                raise ValueError("本地 CLAP 模型不可用；请使用类别查询，不会自动联网下载。")
            import torch
            from transformers import ClapModel, ClapProcessor

            torch.set_num_threads(4)
            self.processor = ClapProcessor.from_pretrained(self.model_path, local_files_only=True)
            self.model = ClapModel.from_pretrained(self.model_path, local_files_only=True).eval()
            self.provenance = {
                "model_directory": self.model_path.name,
                "config_sha256": sha256(self.model_path / "config.json"),
                "weights_sha256": sha256(
                    next(
                        p
                        for p in (
                            self.model_path / "model.safetensors",
                            self.model_path / "pytorch_model.bin",
                        )
                        if p.is_file()
                    )
                ),
            }

    @staticmethod
    def _vector(output):
        tensor = output if hasattr(output, "detach") else output.pooler_output
        array = tensor.detach().cpu().numpy().astype(np.float32)
        return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)

    def search(self, path: Path, query: str, limit=10) -> list[dict]:
        query = query.strip()
        if not query or len(query) > 500:
            raise ValueError("请输入 1–500 字符的英文声音描述。")
        # The bundled CLAP text encoder is English; do not silently claim Chinese support.
        if any("\u4e00" <= c <= "\u9fff" for c in query):
            raise ValueError("CLAP 开放描述请用英文，例如 metal clanging；中文请用类别查询。")
        duration = validate_audio(path)
        if duration > 300:
            raise ValueError("CLAP 试验功能每次最多检索 5 分钟音频，请先裁剪。")
        import torch

        with self.lock, torch.inference_mode():
            self._load()
            key = sha256(path)
            if key != self.index_key:
                samples, sr = read_wav_mono(path, 48000)
                starts = list(range(0, max(1, len(samples) - 10 * sr + 5 * sr), 5 * sr))
                vectors, spans = [], []
                for start in starts:
                    chunk = samples[start : start + 10 * sr]
                    inputs = self.processor(
                        audio=[chunk], sampling_rate=sr, return_tensors="pt", padding=True
                    )
                    inputs = {
                        k: v for k, v in inputs.items() if k in {"input_features", "is_longer"}
                    }
                    vectors.append(self._vector(self.model.get_audio_features(**inputs)))
                    spans.append((start / sr, min(duration, start / sr + 10)))
                embeddings = np.concatenate(vectors)
                if not np.isfinite(embeddings).all():
                    raise ValueError("CLAP 索引包含非有限值。")
                self.embeddings, self.spans, self.index_key = embeddings, spans, key
            text = self.processor(text=[query], return_tensors="pt", padding=True)
            vector = self._vector(self.model.get_text_features(**text))
            scores = (self.embeddings @ vector.T)[:, 0]
            order = np.argsort(-scores, kind="stable")[: max(1, min(int(limit), 20))]
            return [
                {
                    "label": "semantic_candidate",
                    "onset_s": self.spans[i][0],
                    "offset_s": self.spans[i][1],
                    "similarity": float(scores[i]),
                    "query": query,
                    "query_source": "clap_cosine_similarity",
                    "model": self.provenance,
                    "notice": "候选片段，不是检测结论；相似度不是正确率。",
                }
                for i in order
            ]


def label_description() -> str:
    return "、".join(f"{chinese} ({label})" for label, chinese in LABELS_ZH.items())
