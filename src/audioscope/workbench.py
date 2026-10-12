"""Local analysis history and append-only human review, separate from training data."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from sqlalchemy import JSON, Column, Integer, MetaData, String, Table, create_engine, select, update
from sqlalchemy.exc import IntegrityError

from .audio import read_wav_mono

LABELS_ZH = {
    "alarm_bell_ringing": "铃声",
    "blender": "搅拌机",
    "cat": "猫叫",
    "dishes": "餐具碰撞",
    "dog": "狗叫",
    "electric_shaver_toothbrush": "电动剃须刀或牙刷",
    "frying": "煎炸声",
    "running_water": "流水声",
    "speech": "说话声",
    "vacuum_cleaner": "吸尘器",
    "unknown": "待确认声音",
}
metadata = MetaData()
jobs = Table(
    "workbench_jobs",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("created_at", String(40), nullable=False),
    Column("status", String(20), nullable=False),
    Column("revision", Integer, nullable=False, default=0),
    Column("payload", JSON, nullable=False),
)
reviews = Table(
    "workbench_reviews",
    metadata,
    Column("job_id", String(36), primary_key=True),
    Column("revision", Integer, primary_key=True),
    Column("created_at", String(40), nullable=False),
    Column("payload", JSON, nullable=False),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_audio(path: Path, max_seconds=1800, max_bytes=100 * 1024 * 1024) -> float:
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError("音频不存在或超过 100 MiB 限制。")
    try:
        try:
            sample_rate, samples = wavfile.read(path, mmap=True)
        except ValueError:
            # 24-bit PCM cannot be memory mapped; the byte limit bounds this fallback.
            sample_rate, samples = wavfile.read(path, mmap=False)
        if sample_rate <= 0 or samples.ndim not in (1, 2) or not np.isfinite(samples).all():
            raise ValueError("Invalid audio samples")
        duration = len(samples) / sample_rate
    except (ValueError, EOFError) as exc:
        raise ValueError("请上传有效且不含 NaN/Inf 的 PCM 或浮点 WAV 音频。") from exc
    if not 0 < duration <= max_seconds:
        raise ValueError(f"音频时长必须在 0–{max_seconds} 秒之间。")
    return duration


def validate_events(events: list[dict], duration: float) -> list[dict]:
    if len(events) > 10000:
        raise ValueError("事件数量超过 10,000。")
    result = []
    for event in events:
        label = str(event["label"])
        start, end = float(event["onset_s"]), float(event["offset_s"])
        if label not in LABELS_ZH:
            raise ValueError(f"不支持的类别：{label}")
        if not all(map(math.isfinite, (start, end))) or not 0 <= start < end <= duration:
            raise ValueError("每个事件必须满足：0 ≤ 开始 < 结束 ≤ 音频时长。")
        result.append({"label": label, "onset_s": start, "offset_s": end})
    return sorted(result, key=lambda e: (e["onset_s"], e["label"], e["offset_s"]))


class WorkbenchStore:
    """SQLite locally; the same tables can use PostgreSQL after Alembic migration."""

    def __init__(self, root: Path, database_url: str | None = None):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(database_url or f"sqlite:///{self.root / 'history.sqlite3'}")
        if self.engine.dialect.name == "sqlite":
            metadata.create_all(self.engine)
        for folder in ("audio", "exports"):
            (self.root / folder).mkdir(exist_ok=True)

    def create(self, payload: dict) -> str:
        job_id = str(uuid.uuid4())
        with self.engine.begin() as connection:
            connection.execute(
                jobs.insert().values(
                    id=job_id,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    status="running",
                    revision=0,
                    payload=payload,
                )
            )
        return job_id

    def finish(self, job_id: str, payload: dict, status="completed") -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(jobs)
                .where(jobs.c.id == job_id)
                .values(
                    payload=payload,
                    status=status,
                )
            )

    def get(self, job_id: str) -> dict:
        with self.engine.connect() as connection:
            row = connection.execute(select(jobs).where(jobs.c.id == job_id)).mappings().first()
            if row is None:
                raise ValueError("找不到分析任务。")
            result = dict(row)
            result["review"] = None
            if row["revision"]:
                result["review"] = connection.execute(
                    select(reviews.c.payload).where(
                        reviews.c.job_id == job_id,
                        reviews.c.revision == row["revision"],
                    )
                ).scalar_one()
            return result

    def history(self, limit=100) -> list[dict]:
        with self.engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    select(jobs).order_by(jobs.c.created_at.desc()).limit(min(limit, 100))
                ).mappings()
            ]

    def save_review(
        self, job_id: str, events: list[dict], expected_revision: int, confirmed: bool
    ) -> int:
        if not confirmed:
            raise ValueError("请先确认这些是你已复核的标签；系统不会将其自动用于训练。")
        job = self.get(job_id)
        if job["status"] != "completed":
            raise ValueError("只能复核已完成的任务。")
        checked = validate_events(events, job["payload"]["duration_s"])
        revision = expected_revision + 1
        try:
            with self.engine.begin() as connection:
                changed = connection.execute(
                    update(jobs)
                    .where(
                        jobs.c.id == job_id,
                        jobs.c.revision == expected_revision,
                    )
                    .values(revision=revision)
                )
                if changed.rowcount != 1:
                    raise ValueError("结果已在其他会话修改，请重新打开任务后再保存。")
                connection.execute(
                    reviews.insert().values(
                        job_id=job_id,
                        revision=revision,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        payload={
                            "events": checked,
                            "source": (
                                "software_verification_fixture"
                                if job["payload"].get("verification_fixture")
                                else "human_review"
                            ),
                            "training_eligible": False,
                            "confirmed": True,
                        },
                    )
                )
        except IntegrityError as exc:
            raise ValueError("复核版本冲突，请重新打开任务。") from exc
        return revision

    def retained_audio(self, job_id: str) -> Path | None:
        job = self.get(job_id)
        path = self.root / "audio" / f"{job['id']}.wav"
        return path if job["payload"].get("retain_audio") and path.is_file() else None

    def export(self, job_id: str) -> list[str]:
        job = self.get(job_id)
        if job["status"] != "completed":
            raise ValueError("分析尚未完成，不能导出。")
        directory = self.root / "exports" / str(uuid.uuid4())
        directory.mkdir()
        document = {**job, "notice": "预测及人工复核分开保存；均不自动用于训练。"}
        (directory / "analysis.json").write_text(
            json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        columns = ["source", "label", "onset_s", "offset_s", "confidence", "mean_confidence"]
        with (directory / "events.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for e in job["payload"]["events"]:
                writer.writerow({**e, "source": "model_prediction"})
            for e in (job["review"] or {}).get("events", []):
                writer.writerow({**e, "source": job["review"]["source"]})
        return [str(directory / name) for name in ("analysis.json", "events.csv")]

    def cleanup_exports(self, now=None, ttl_seconds=86400) -> int:
        """Delete only generated UUID directories older than the explicit TTL; no raw audio."""
        cutoff = (time.time() if now is None else now) - ttl_seconds
        removed = 0
        for path in (self.root / "exports").iterdir():
            try:
                uuid.UUID(path.name)
            except ValueError:
                continue
            if path.is_dir() and not path.is_symlink() and path.stat().st_mtime < cutoff:
                shutil.rmtree(path)
                removed += 1
        return removed


class AnalysisService:
    def __init__(self, deployment, store: WorkbenchStore):
        self.deployment, self.store = deployment, store

    def analyse(self, path: Path, profile="overall", retain_audio=False) -> dict:
        duration = validate_audio(path)
        payload = {
            "audio_sha256": sha256(path),
            "duration_s": duration,
            "threshold_profile": profile,
            "retain_audio": bool(retain_audio),
            "model": self.deployment.provenance,
            "training_eligible": False,
        }
        job_id = self.store.create(payload)
        try:
            samples, _ = read_wav_mono(path, self.deployment.sample_rate)
            if not np.isfinite(samples).all():
                raise ValueError("音频包含非有限采样值。")
            result = self.deployment.predict_samples(samples, threshold_profile=profile)
            payload.update(
                {
                    "events": sorted(
                        result["events"] + result["unknown_events"],
                        key=lambda e: (e["onset_s"], e["label"]),
                    ),
                    "stats": result["stats"],
                    "backend": result["backend"],
                }
            )
            if retain_audio:
                shutil.copyfile(path, self.store.root / "audio" / f"{job_id}.wav")
            self.store.finish(job_id, payload)
        except Exception as exc:
            self.store.finish(job_id, {**payload, "error_type": type(exc).__name__}, "failed")
            raise
        return self.store.get(job_id)


def summary_zh(job: dict) -> str:
    payload = job["payload"]
    known = [e for e in payload["events"] if e["label"] != "unknown"]
    labels = {e["label"] for e in known}
    unknown = len(payload["events"]) - len(known)
    return (
        f"音频 **{payload['duration_s']:.2f} 秒**；预测 **{len(labels)} 类已知声音、"
        f"{len(known)} 段事件**，另有 **{unknown} 段待确认声音**。\n\n"
        "得分不是正确率：最高分是事件内峰值，平均分反映片段整体。"
        "unknown 不等于静音，也不表示已识别出新类别。\n\n"
        f"任务 `{job['id']}` · 复核版本 {job['revision']} · 原音频"
        f"{'已按你的选择保留' if payload['retain_audio'] else '未永久保留'}。"
    )


def playback(path: Path, event: dict, context_s=0.5):
    validate_audio(path)
    samples, sr = read_wav_mono(path, 16000)
    validate_events([{**event, "label": "unknown"}], len(samples) / sr)
    left = max(0, int((event["onset_s"] - context_s) * sr))
    right = min(len(samples), int((event["offset_s"] + context_s) * sr))
    return sr, samples[left:right]
