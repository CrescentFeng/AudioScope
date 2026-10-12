from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

SCHEMA = "catalog"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Source(Base):
    __tablename__ = "sources"
    __table_args__ = (UniqueConstraint("name", "source_url"), {"schema": SCHEMA})

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200))
    source_url: Mapped[str] = mapped_column(Text)
    citation: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class License(Base):
    __tablename__ = "licenses"
    __table_args__ = (UniqueConstraint("spdx_or_name"), {"schema": SCHEMA})

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    spdx_or_name: Mapped[str] = mapped_column(String(160))
    license_url: Mapped[Optional[str]] = mapped_column(Text)
    commercial_use_allowed: Mapped[Optional[bool]] = mapped_column(Boolean)
    notes: Mapped[Optional[str]] = mapped_column(Text)


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("slug"), {"schema": SCHEMA})

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(100))
    display_name: Mapped[str] = mapped_column(String(200))
    task_type: Mapped[str] = mapped_column(String(80))
    source_id: Mapped[int] = mapped_column(ForeignKey(f"{SCHEMA}.sources.id"))
    license_id: Mapped[int] = mapped_column(ForeignKey(f"{SCHEMA}.licenses.id"))
    description: Mapped[Optional[str]] = mapped_column(Text)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (
        UniqueConstraint("dataset_id", "version_key"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey(f"{SCHEMA}.datasets.id"))
    version_key: Mapped[str] = mapped_column(String(160))
    manifest_uri: Mapped[str] = mapped_column(Text)
    manifest_sha256: Mapped[str] = mapped_column(String(64))
    record_count: Mapped[int] = mapped_column(Integer)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AudioFile(Base):
    __tablename__ = "audio_files"
    __table_args__ = (
        CheckConstraint("duration_s > 0", name="ck_audio_duration_positive"),
        CheckConstraint("sample_rate > 0", name="ck_audio_sample_rate_positive"),
        CheckConstraint("channels > 0", name="ck_audio_channels_positive"),
        UniqueConstraint("dataset_version_id", "storage_uri"),
        Index("ix_audio_files_source_recording", "source_recording_id"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_version_id: Mapped[str] = mapped_column(
        ForeignKey(f"{SCHEMA}.dataset_versions.id", ondelete="CASCADE")
    )
    storage_uri: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64))
    byte_size: Mapped[Optional[int]] = mapped_column(BigInteger)
    duration_s: Mapped[float] = mapped_column(Float)
    sample_rate: Mapped[int] = mapped_column(Integer)
    channels: Mapped[int] = mapped_column(Integer)
    codec: Mapped[str] = mapped_column(String(40), default="pcm_s16le")
    source_recording_id: Mapped[Optional[str]] = mapped_column(String(240))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Annotation(Base):
    __tablename__ = "annotations"
    __table_args__ = (
        CheckConstraint("onset_s >= 0", name="ck_annotation_onset_nonnegative"),
        CheckConstraint("offset_s >= onset_s", name="ck_annotation_offset_after_onset"),
        Index("ix_annotations_label", "label"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    audio_file_id: Mapped[str] = mapped_column(
        ForeignKey(f"{SCHEMA}.audio_files.id", ondelete="CASCADE")
    )
    label: Mapped[str] = mapped_column(String(160))
    onset_s: Mapped[float] = mapped_column(Float)
    offset_s: Mapped[float] = mapped_column(Float)
    annotation_type: Mapped[str] = mapped_column(String(40))
    annotator: Mapped[Optional[str]] = mapped_column(String(160))
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class DatasetSplit(Base):
    __tablename__ = "dataset_splits"
    __table_args__ = (
        UniqueConstraint("audio_file_id", "split_name"),
        Index("ix_dataset_splits_name", "dataset_version_id", "split_name"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_version_id: Mapped[str] = mapped_column(
        ForeignKey(f"{SCHEMA}.dataset_versions.id", ondelete="CASCADE")
    )
    audio_file_id: Mapped[str] = mapped_column(
        ForeignKey(f"{SCHEMA}.audio_files.id", ondelete="CASCADE")
    )
    split_name: Mapped[str] = mapped_column(String(80))
    protocol: Mapped[str] = mapped_column(String(200))


class ProcessingRun(Base):
    __tablename__ = "processing_runs"
    __table_args__ = (Index("ix_processing_runs_kind", "run_kind"), {"schema": SCHEMA})

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_kind: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(40))
    config_uri: Mapped[Optional[str]] = mapped_column(Text)
    git_commit: Mapped[Optional[str]] = mapped_column(String(64))
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    environment_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    summary_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"
    __table_args__ = (
        UniqueConstraint("storage_uri", "sha256"),
        Index("ix_model_artifacts_stage", "stage"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[Optional[str]] = mapped_column(ForeignKey(f"{SCHEMA}.processing_runs.id"))
    name: Mapped[str] = mapped_column(String(200))
    stage: Mapped[str] = mapped_column(String(40))
    format: Mapped[str] = mapped_column(String(40))
    storage_uri: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64))
    byte_size: Mapped[int] = mapped_column(BigInteger)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"
    __table_args__ = (Index("ix_benchmark_results_metric", "metric_name"), {"schema": SCHEMA})

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[Optional[str]] = mapped_column(ForeignKey(f"{SCHEMA}.processing_runs.id"))
    model_artifact_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey(f"{SCHEMA}.model_artifacts.id")
    )
    metric_name: Mapped[str] = mapped_column(String(160))
    metric_value: Mapped[float] = mapped_column(Float)
    unit: Mapped[Optional[str]] = mapped_column(String(40))
    slice_name: Mapped[Optional[str]] = mapped_column(String(160))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class FieldSession(Base):
    __tablename__ = "field_sessions"
    __table_args__ = (Index("ix_field_sessions_status", "status"), {"schema": SCHEMA})

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    device_name: Mapped[str] = mapped_column(String(240))
    room_name: Mapped[str] = mapped_column(String(240))
    distance_m: Mapped[Optional[float]] = mapped_column(Float)
    playback_level_db: Mapped[Optional[float]] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(40), default="planned")
    operator_notes: Mapped[Optional[str]] = mapped_column(Text)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class FieldRecording(Base):
    __tablename__ = "field_recordings"
    __table_args__ = (UniqueConstraint("field_session_id", "storage_uri"), {"schema": SCHEMA})

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    field_session_id: Mapped[str] = mapped_column(
        ForeignKey(f"{SCHEMA}.field_sessions.id", ondelete="CASCADE")
    )
    storage_uri: Mapped[str] = mapped_column(Text)
    sha256: Mapped[Optional[str]] = mapped_column(String(64))
    duration_s: Mapped[Optional[float]] = mapped_column(Float)
    sample_rate: Mapped[Optional[int]] = mapped_column(Integer)
    channels: Mapped[Optional[int]] = mapped_column(Integer)
    sync_offset_s: Mapped[Optional[float]] = mapped_column(Float)
    captured_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class QualityCheck(Base):
    __tablename__ = "quality_checks"
    __table_args__ = (
        Index("ix_quality_checks_scope", "scope_type", "scope_id"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    scope_type: Mapped[str] = mapped_column(String(80))
    scope_id: Mapped[str] = mapped_column(String(64))
    check_name: Mapped[str] = mapped_column(String(160))
    status: Mapped[str] = mapped_column(String(40))
    observed_value: Mapped[Optional[float]] = mapped_column(Float)
    threshold_value: Mapped[Optional[float]] = mapped_column(Float)
    details_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReleaseAudit(Base):
    __tablename__ = "release_audits"
    __table_args__ = (Index("ix_release_audits_created", "created_at"), {"schema": SCHEMA})

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    mode: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(40))
    report_uri: Mapped[str] = mapped_column(Text)
    checks_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def database_url(explicit: Optional[str] = None) -> str:
    value = explicit or os.getenv("AUDIOSCOPE_DATABASE_URL")
    if not value:
        raise RuntimeError("Set AUDIOSCOPE_DATABASE_URL or pass --database-url")
    return value


def make_engine(explicit_url: Optional[str] = None):
    return create_engine(database_url(explicit_url), pool_pre_ping=True)


def make_session_factory(explicit_url: Optional[str] = None):
    return sessionmaker(make_engine(explicit_url), expire_on_commit=False)


def readiness(explicit_url: Optional[str] = None) -> dict[str, Any]:
    engine = make_engine(explicit_url)
    with engine.connect() as connection:
        value = connection.execute(text("SELECT 1")).scalar_one()
        revision = connection.execute(
            text("SELECT version_num FROM public.alembic_version LIMIT 1")
        ).scalar_one_or_none()
    return {"database": "ready" if value == 1 else "unavailable", "revision": revision}
