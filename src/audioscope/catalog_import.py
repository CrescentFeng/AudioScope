from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import func, select

from .db import (
    Annotation,
    AudioFile,
    BenchmarkResult,
    Dataset,
    DatasetSplit,
    DatasetVersion,
    FieldSession,
    License,
    ModelArtifact,
    ProcessingRun,
    Source,
    make_session_factory,
)

NAMESPACE = uuid.UUID("9be12ec8-87a2-4d40-858e-9e2786c7cf28")


def _stable_id(*parts: object) -> str:
    return str(uuid.uuid5(NAMESPACE, "|".join(str(part) for part in parts)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _get_or_create(session, model, defaults: dict[str, Any], **lookup: Any):
    instance = session.scalar(select(model).filter_by(**lookup))
    if instance is None:
        instance = model(**lookup, **defaults)
        session.add(instance)
        session.flush()
    else:
        for key, value in defaults.items():
            setattr(instance, key, value)
    return instance


def _upsert_dataset(
    session,
    slug: str,
    display_name: str,
    task_type: str,
    source_name: str,
    source_url: str,
    license_name: str,
    commercial_use_allowed: bool | None,
    description: str,
):
    source = _get_or_create(
        session,
        Source,
        {"citation": source_url},
        name=source_name,
        source_url=source_url,
    )
    license_row = _get_or_create(
        session,
        License,
        {
            "license_url": None,
            "commercial_use_allowed": commercial_use_allowed,
            "notes": None,
        },
        spdx_or_name=license_name,
    )
    return _get_or_create(
        session,
        Dataset,
        {
            "display_name": display_name,
            "task_type": task_type,
            "source_id": source.id,
            "license_id": license_row.id,
            "description": description,
        },
        slug=slug,
    )


def _version(
    session,
    dataset,
    version_key: str,
    manifest_path: Path,
    root: Path,
    record_count: int,
    metadata: dict[str, Any],
) -> DatasetVersion:
    row = DatasetVersion(
        id=_stable_id(dataset.slug, version_key),
        dataset_id=dataset.id,
        version_key=version_key,
        manifest_uri=f"audioscope://{manifest_path.relative_to(root).as_posix()}",
        manifest_sha256=_sha256(manifest_path),
        record_count=record_count,
        metadata_json=metadata,
        imported_at=datetime.now(timezone.utc),
    )
    session.merge(row)
    session.flush()
    return row


def _wav_info(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as handle:
        return handle.getframerate(), handle.getnchannels(), handle.getnframes()


def import_esc50(session, root: Path, hash_files: bool) -> dict[str, int]:
    manifest = root / "data/manifests/esc50.jsonl"
    records = _jsonl(manifest)
    version_key = str(records[0]["dataset_version"])
    dataset = _upsert_dataset(
        session,
        "esc50",
        "ESC-50",
        "environmental_sound_classification",
        "ESC-50 GitHub release",
        str(records[0]["source_url"]),
        str(records[0]["license"]),
        False,
        "2,000 five-second environmental recordings in 50 classes and five official folds.",
    )
    version = _version(
        session,
        dataset,
        version_key,
        manifest,
        root,
        len(records),
        {"split_protocol": "official five-fold cross-validation"},
    )
    audio_root = root / "data/raw/ESC-50-master"
    for item in records:
        local_path = audio_root / item["path"]
        file_id = str(item["asset_id"])
        file_hash = _sha256(local_path) if hash_files else str(item["sha256"])
        session.merge(
            AudioFile(
                id=file_id,
                dataset_version_id=version.id,
                storage_uri=f"audioscope-data://raw/ESC-50-master/{item['path']}",
                sha256=file_hash,
                byte_size=local_path.stat().st_size,
                duration_s=float(item["duration_s"]),
                sample_rate=int(item["sample_rate"]),
                channels=int(item["channels"]),
                codec="pcm_wav",
                source_recording_id=str(item["source_recording_id"]),
                metadata_json={"original_path": item["path"]},
            )
        )
        session.merge(
            Annotation(
                id=_stable_id(file_id, "clip", item["label"]),
                audio_file_id=file_id,
                label=str(item["label"]),
                onset_s=0.0,
                offset_s=float(item["duration_s"]),
                annotation_type="clip_level",
                annotator="ESC-50",
                confidence=1.0,
                metadata_json={},
            )
        )
        session.merge(
            DatasetSplit(
                id=_stable_id(file_id, f"fold-{item['fold']}"),
                dataset_version_id=version.id,
                audio_file_id=file_id,
                split_name=f"fold-{int(item['fold'])}",
                protocol="official ESC-50 fold",
            )
        )
    session.commit()
    return {"audio_files": len(records), "annotations": len(records), "splits": len(records)}


def import_desed(session, root: Path, hash_files: bool) -> dict[str, int]:
    dataset_root = root / "data/desed-real-strong-only"
    manifest = dataset_root / "manifest.jsonl"
    metadata = json.loads((dataset_root / "metadata.json").read_text(encoding="utf-8"))
    records = _jsonl(manifest)
    dataset = _upsert_dataset(
        session,
        "desed-real-strong-only",
        str(metadata["dataset"]),
        "strongly_labeled_sound_event_detection",
        "DESED real strong WebDataset repack",
        str(metadata["source_record"]),
        str(metadata["license"]),
        True,
        "Strongly labelled real domestic soundscapes with source-disjoint derived splits.",
    )
    version = _version(
        session,
        dataset,
        str(metadata["signature"]),
        manifest,
        root,
        len(records),
        metadata,
    )
    annotation_count = 0
    for item in records:
        local_path = dataset_root / item["path"]
        sample_rate, channels, frames = _wav_info(local_path)
        file_id = _stable_id(version.id, item["scene_id"])
        session.merge(
            AudioFile(
                id=file_id,
                dataset_version_id=version.id,
                storage_uri=f"audioscope-data://desed-real-strong-only/{item['path']}",
                sha256=_sha256(local_path) if hash_files else _stable_id("unhashed", item["path"]),
                byte_size=local_path.stat().st_size,
                duration_s=frames / sample_rate,
                sample_rate=sample_rate,
                channels=channels,
                codec="pcm_wav",
                source_recording_id=str(item["scene_id"]),
                metadata_json={"original_path": item["path"]},
            )
        )
        for index, event in enumerate(item["events"]):
            annotation_count += 1
            session.merge(
                Annotation(
                    id=_stable_id(
                        file_id, index, event["label"], event["onset_s"], event["offset_s"]
                    ),
                    audio_file_id=file_id,
                    label=str(event["label"]),
                    onset_s=float(event["onset_s"]),
                    offset_s=float(event["offset_s"]),
                    annotation_type="strong",
                    annotator="DESED",
                    confidence=1.0,
                    metadata_json={
                        "source_asset_id": event.get("source_asset_id"),
                        "source_recording_id": event.get("source_recording_id"),
                    },
                )
            )
        session.merge(
            DatasetSplit(
                id=_stable_id(file_id, item["split"]),
                dataset_version_id=version.id,
                audio_file_id=file_id,
                split_name=str(item["split"]),
                protocol=str(metadata["split_protocol"]),
            )
        )
    session.commit()
    return {"audio_files": len(records), "annotations": annotation_count, "splits": len(records)}


def _iter_model_files(root: Path) -> Iterable[Path]:
    for pattern in ("*.pt", "*.onnx", "*.joblib"):
        yield from sorted((root / "artifacts").rglob(pattern))


def import_l4_artifacts(session, root: Path) -> dict[str, int]:
    summary_path = root / "reports/generated/l4-engineering/l4_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_id = _stable_id("l4-engineering", _sha256(summary_path))
    run = ProcessingRun(
        id=run_id,
        run_kind="l4_engineering",
        status="completed",
        config_uri="audioscope://configs/l4_engineering.yaml",
        git_commit=_git_commit(root),
        started_at=None,
        finished_at=datetime.fromtimestamp(summary_path.stat().st_mtime, timezone.utc),
        environment_json={"runtime": "local CPU"},
        summary_json={"protocol": summary.get("protocol")},
    )
    session.merge(run)
    model_count = 0
    artifact_ids: dict[str, str] = {}
    for path in _iter_model_files(root):
        relative = path.relative_to(root).as_posix()
        digest = _sha256(path)
        model_id = _stable_id(relative, digest)
        artifact_ids[path.name] = model_id
        session.merge(
            ModelArtifact(
                id=model_id,
                run_id=run_id if "l4-engineering" in relative else None,
                name=path.name,
                stage=relative.split("/")[1] if "/" in relative else "unknown",
                format=path.suffix.lstrip("."),
                storage_uri=f"audioscope-artifact://{relative.removeprefix('artifacts/')}",
                sha256=digest,
                byte_size=path.stat().st_size,
                metadata_json={},
                created_at=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc),
            )
        )
        model_count += 1
    metric_count = 0
    for csv_name in ("benchmark.csv", "snr_robustness.csv"):
        with (summary_path.parent / csv_name).open(encoding="utf-8", newline="") as handle:
            for row_index, row in enumerate(csv.DictReader(handle)):
                slice_name = row.get("backend") or f"snr={row.get('snr_db')}"
                model_id = artifact_ids.get(
                    "crnn_short_event_dynamic_int8.onnx"
                    if row.get("backend") == "onnxruntime_dynamic_int8"
                    else "crnn_short_event_fp32.onnx"
                    if row.get("backend") == "onnxruntime_fp32"
                    else "crnn_short_event_calibrated.pt"
                )
                for key, raw_value in row.items():
                    if key in {"backend", "snr_db", "clips"} or raw_value in {None, ""}:
                        continue
                    try:
                        value = float(raw_value)
                    except ValueError:
                        continue
                    metric_id = _stable_id(run_id, csv_name, row_index, key)
                    unit = "ms" if "latency_ms" in key else "bytes" if "bytes" in key else None
                    session.merge(
                        BenchmarkResult(
                            id=metric_id,
                            run_id=run_id,
                            model_artifact_id=model_id,
                            metric_name=key,
                            metric_value=value,
                            unit=unit,
                            slice_name=slice_name,
                            metadata_json={"source_csv": csv_name},
                        )
                    )
                    metric_count += 1
    session.commit()
    return {"model_artifacts": model_count, "benchmark_results": metric_count, "runs": 1}


def import_field_plan(session, root: Path) -> dict[str, int]:
    matrix = root / "data/l4-field-kit/recording_matrix.csv"
    if not matrix.exists():
        return {"field_sessions": 0}
    with matrix.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        session.merge(
            FieldSession(
                id=str(row["session_id"]),
                device_name=str(row["device_name"]),
                room_name=str(row["room_name"]),
                distance_m=float(row["distance_m"]) if row["distance_m"] else None,
                playback_level_db=(
                    float(row["playback_level_db"]) if row["playback_level_db"] else None
                ),
                status=str(row["status"]),
                operator_notes=str(row["operator_notes"]) or None,
                started_at=None,
                metadata_json={"planned_recording_path": row["recording_path"]},
            )
        )
    session.commit()
    return {"field_sessions": len(rows)}


def verify_catalog(session) -> dict[str, Any]:
    models = {
        "datasets": Dataset,
        "dataset_versions": DatasetVersion,
        "audio_files": AudioFile,
        "annotations": Annotation,
        "dataset_splits": DatasetSplit,
        "processing_runs": ProcessingRun,
        "model_artifacts": ModelArtifact,
        "benchmark_results": BenchmarkResult,
        "field_sessions": FieldSession,
    }
    counts = {
        name: int(session.scalar(select(func.count()).select_from(model)) or 0)
        for name, model in models.items()
    }
    expected_minimums = {
        "datasets": 2,
        "audio_files": 3153,
        "annotations": 2000 + 4224,
        "dataset_splits": 3153,
        "model_artifacts": 1,
        "benchmark_results": 1,
    }
    checks = {
        name: {"observed": counts[name], "minimum": minimum, "passed": counts[name] >= minimum}
        for name, minimum in expected_minimums.items()
    }
    return {
        "status": "passed" if all(row["passed"] for row in checks.values()) else "failed",
        "counts": counts,
        "checks": checks,
    }


def run_import(root: Path, database_url: str | None, hash_files: bool) -> dict[str, Any]:
    session_factory = make_session_factory(database_url)
    with session_factory() as session:
        result = {
            "esc50": import_esc50(session, root, hash_files),
            "desed": import_desed(session, root, hash_files),
            "l4": import_l4_artifacts(session, root),
            "field_plan": import_field_plan(session, root),
        }
        result["verification"] = verify_catalog(session)
        result["status"] = result["verification"]["status"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Import AudioScope metadata into PostgreSQL")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--database-url")
    parser.add_argument("--hash-files", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.verify_only:
        with make_session_factory(args.database_url)() as session:
            result = verify_catalog(session)
    else:
        result = run_import(args.root.resolve(), args.database_url, args.hash_files)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") == "failed" or result.get("verification", {}).get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
