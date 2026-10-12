"""Trusted single-user analysis API, mounted alongside the read-only catalog endpoints."""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .model_registry import resolve_model
from .retrieval import ClapSearch, query_events
from .workbench import AnalysisService, WorkbenchStore

router = APIRouter(prefix="/v1/analysis", tags=["analysis workbench"])
_lock = threading.Lock()
_service = None
_searcher = None
_model_services = {}


def service(model_key="current"):
    global _service, _searcher
    if _service is None:
        from .engineering import SEDDeployment

        root = Path(os.getenv("AUDIOSCOPE_ROOT", ".")).resolve()
        model, onnx = resolve_model(root)
        store = WorkbenchStore(root / "runtime/workbench", os.environ["AUDIOSCOPE_DATABASE_URL"])
        _service = AnalysisService(SEDDeployment(model, onnx), store)
        _searcher = ClapSearch(root / "data/pretrained/clap-htsat-unfused-8fa0f1c")
    if model_key == "current":
        return _service
    if model_key not in _model_services:
        from .engineering import SEDDeployment

        root = Path(os.getenv("AUDIOSCOPE_ROOT", ".")).resolve()
        model, onnx = resolve_model(root, model_key)
        _model_services[model_key] = AnalysisService(SEDDeployment(model, onnx), _service.store)
    return _model_services[model_key]


@router.post("")
def analyse(
    audio: UploadFile = File(...),
    profile: str = Form("overall"),
    retain_audio: bool = Form(False),
    model_key: str = Form("current"),
):
    if profile not in {"overall", "short_event", "class_balanced"}:
        raise HTTPException(422, "Unknown threshold profile")
    try:
        with tempfile.TemporaryDirectory(prefix="audioscope-upload-") as folder:
            path = Path(folder) / "audio.wav"
            total = 0
            with path.open("wb") as output:
                while chunk := audio.file.read(1024 * 1024):
                    total += len(chunk)
                    if total > 100 * 1024 * 1024:
                        raise HTTPException(413, "Maximum upload size is 100 MiB")
                    output.write(chunk)
            with _lock:
                return service(model_key).analyse(path, profile, retain_audio)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"Analysis unavailable: {type(exc).__name__}") from exc
    finally:
        audio.file.close()


@router.get("")
def history():
    with _lock:
        return service().store.history()


@router.get("/{job_id}")
def detail(job_id: str):
    try:
        with _lock:
            return service().store.get(job_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


class ReviewRequest(BaseModel):
    events: list[dict] = Field(max_length=10000)
    expected_revision: int = Field(ge=0)
    confirmed: bool = False


@router.post("/{job_id}/reviews")
def review(job_id: str, request: ReviewRequest):
    try:
        with _lock:
            version = service().store.save_review(
                job_id, request.events, request.expected_revision, request.confirmed
            )
            return {"revision": version, "training_eligible": False}
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    semantic: bool = False
    require_overlap: bool = False


@router.post("/{job_id}/search")
def search(job_id: str, request: SearchRequest):
    try:
        with _lock:
            store = service().store
            job = store.get(job_id)
            if job["status"] != "completed":
                raise ValueError("任务没有完整结果。")
            if not request.semantic:
                return query_events(
                    job["payload"]["events"], request.query, request.require_overlap
                )
            path = store.retained_audio(job_id)
            if path is None:
                raise ValueError("这个任务未保留原音频，不能重新生成 CLAP 索引。")
            return _searcher.search(path, request.query)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
