from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import func, select

from .db import (
    AudioFile,
    BenchmarkResult,
    Dataset,
    DatasetVersion,
    ModelArtifact,
    make_session_factory,
    readiness,
)
from .workbench_api import router as workbench_router


def _url() -> str:
    value = os.getenv("AUDIOSCOPE_DATABASE_URL")
    if not value:
        raise RuntimeError("AUDIOSCOPE_DATABASE_URL is required")
    return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.sessions = make_session_factory(_url())
    yield


app = FastAPI(
    title="AudioScope Catalog API",
    version="1.0.0",
    description="Local single-user catalog, analysis and human-review workbench for AudioScope.",
    lifespan=lifespan,
)
app.include_router(workbench_router)


@app.get("/health")
def health() -> dict:
    try:
        return {"status": "ok", **readiness(_url())}
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"database unavailable: {type(error).__name__}")


@app.get("/v1/readiness")
def catalog_readiness() -> dict:
    with app.state.sessions() as session:
        counts = {
            "datasets": session.scalar(select(func.count()).select_from(Dataset)),
            "audio_files": session.scalar(select(func.count()).select_from(AudioFile)),
            "models": session.scalar(select(func.count()).select_from(ModelArtifact)),
            "benchmarks": session.scalar(select(func.count()).select_from(BenchmarkResult)),
        }
    ready = counts["datasets"] >= 2 and counts["audio_files"] >= 3153 and counts["models"] >= 1
    return {"status": "ready" if ready else "incomplete", "counts": counts}


@app.get("/v1/datasets")
def datasets() -> list[dict]:
    with app.state.sessions() as session:
        rows = session.execute(
            select(
                Dataset.slug,
                Dataset.display_name,
                Dataset.task_type,
                DatasetVersion.version_key,
                DatasetVersion.record_count,
                DatasetVersion.manifest_sha256,
            ).join(DatasetVersion, DatasetVersion.dataset_id == Dataset.id)
        ).all()
    return [row._asdict() for row in rows]


@app.get("/v1/models")
def models(limit: int = Query(100, ge=1, le=500)) -> list[dict]:
    with app.state.sessions() as session:
        rows = session.scalars(
            select(ModelArtifact).order_by(ModelArtifact.stage, ModelArtifact.name).limit(limit)
        ).all()
    return [
        {
            "id": row.id,
            "name": row.name,
            "stage": row.stage,
            "format": row.format,
            "storage_uri": row.storage_uri,
            "sha256": row.sha256,
            "byte_size": row.byte_size,
        }
        for row in rows
    ]


@app.get("/v1/benchmarks")
def benchmarks(
    metric: str | None = None,
    limit: int = Query(200, ge=1, le=1000),
) -> list[dict]:
    statement = select(BenchmarkResult).order_by(BenchmarkResult.metric_name)
    if metric:
        statement = statement.where(BenchmarkResult.metric_name == metric)
    with app.state.sessions() as session:
        rows = session.scalars(statement.limit(limit)).all()
    return [
        {
            "metric": row.metric_name,
            "value": row.metric_value,
            "unit": row.unit,
            "slice": row.slice_name,
            "model_artifact_id": row.model_artifact_id,
        }
        for row in rows
    ]
