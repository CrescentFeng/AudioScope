from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check(name: str, passed: bool, evidence: str, manual: bool = False) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence, "manual": manual}


def _json_status(path: Path, expected: str | tuple[str, ...]) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("status")
        return value in ((expected,) if isinstance(expected, str) else expected)
    except (json.JSONDecodeError, OSError):
        return False


def _load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def software_checks(root):
    directory = root / "reports/generated/software-release"
    robust, soak, video, catalog = [
        _load(directory / f"{name}.json")
        for name in ("robustness", "stream_soak", "video_validation", "catalog_sync")
    ]
    model = root / "artifacts/l4-engineering/crnn_short_event_calibrated.pt"
    onnx = root / "artifacts/l4-engineering/crnn_short_event_fp32.onnx"
    code = _load(directory / "code_verification.json")
    manifest = root / "data/desed-real-strong-only/manifest.jsonl"

    def current(report):
        return (
            model.is_file()
            and onnx.is_file()
            and report.get("model_sha256") == _sha256(model)
            and report.get("onnx_sha256") == _sha256(onnx)
        )

    def sources_match(report):
        return bool(report.get("source_sha256")) and all(
            (root / path).is_file() and _sha256(root / path) == digest
            for path, digest in report.get("source_sha256", {}).items()
        )

    expected = {"clean", "reverb_025", "reverb_080", "telephone", "laptop", "pink_10db", "combined"}
    rows = robust.get("conditions", [])
    robust_ok = (
        robust.get("status") == "passed"
        and current(robust)
        and robust.get("seed") == 2027
        and robust.get("physical_validation_performed") is False
        and manifest.is_file()
        and robust.get("manifest_sha256") == _sha256(manifest)
        and robust.get("test_clips") == 221
        and len(rows) == 7
        and {r.get("condition") for r in rows} == expected
        and all(
            r.get("clips") == 221
            and r.get("reproducible") is True
            and r.get("prediction_sha256") == r.get("repeat_prediction_sha256")
            and all(
                isinstance(r.get(k), (float, int)) and math.isfinite(r[k])
                for k in ("event_f1", "macro_event_f1", "segment_f1", "model_rtf")
            )
            for r in rows
        )
    )
    soak_ok = (
        soak.get("status") == "passed"
        and current(soak)
        and soak.get("processed_seconds", 0) >= 1800
        and soak.get("rtf", 1) < 0.1
        and soak.get("compute_p95_ms", 1000) < 1000
        and soak.get("rss_growth_bytes_after_warmup", 1e12) <= 64 * 1024**2
        and len(soak.get("checks", {})) >= 8
        and all(soak["checks"].values())
    )
    video_path = root / video.get("video_uri", "missing")
    video_ok = (
        video.get("status") == "passed"
        and video.get("smoke") is False
        and video.get("voice") == "Tingting"
        and 120 <= video.get("duration_s", 0) <= 210
        and video_path.is_file()
        and video.get("video_sha256") == _sha256(video_path)
        and bool(video.get("checks"))
        and all(video["checks"].values())
        and video.get("checks", {}).get("audible_audio") is True
        and bool(video.get("asset_sha256"))
        and all(
            (root / p).is_file() and _sha256(root / p) == digest
            for p, digest in video.get("asset_sha256", {}).items()
        )
        and sources_match(video)
    )
    docs = [root / "docs/PROJECT_SUMMARY_BILINGUAL.md", root / "docs/L4_MODEL_CARD.md"]
    docs_ok = all(
        p.exists()
        and "未进行自有物理设备/房间认证" in p.read_text()
        and "Physical device and room certification has not been performed" in p.read_text()
        for p in docs
    )
    return [
        _check(
            "current code verification",
            code.get("status") == "passed" and sources_match(code),
            str(directory / "code_verification.json"),
        ),
        _check("software domain shifts", robust_ok, str(directory / "robustness.json")),
        _check("bounded 1800-second stream", soak_ok, str(directory / "stream_soak.json")),
        _check("automatic evidence video", video_ok, str(video_path)),
        _check(
            "software release catalog",
            catalog.get("status") == "passed" and sources_match(catalog),
            str(directory / "catalog_sync.json"),
        ),
        _check("physical-validation disclosure", docs_ok, "Bilingual summary and model card"),
    ]


def audit(root: Path, mode: str) -> dict[str, Any]:
    if mode not in {"preflight", "final", "field-final", "software-final"}:
        raise ValueError(mode)
    root = root.resolve()
    database_report = root / "reports/generated/database/import_report.json"
    docker_validation = root / "reports/generated/docker/validation.json"
    field_summary = root / "data/l4-field-kit/field_kit_summary.json"
    simulation = root / "reports/generated/field/mic_simulation.json"
    field_qc = root / "reports/generated/field/field_qc.json"
    live_mic = root / "reports/generated/field/mic_live_acceptance.json"
    non_field = [
        _check(
            "database migration",
            (root / "migrations/versions/0001_audioscope_catalog.py").exists(),
            "Alembic revision 0001",
        ),
        _check("database import", _json_status(database_report, "passed"), str(database_report)),
        _check("Docker Compose", _json_status(docker_validation, "passed"), str(docker_validation)),
        _check(
            "catalog API",
            (root / "src/audioscope/api.py").exists(),
            "health/readiness/catalog endpoints",
        ),
        _check("field kit", _json_status(field_summary, "prepared"), str(field_summary)),
        _check("stream simulation", _json_status(simulation, "passed"), str(simulation)),
        _check(
            "bilingual summary",
            (root / "docs/PROJECT_SUMMARY_BILINGUAL.md").exists(),
            "Chinese and English portfolio copy",
        ),
        _check(
            "video script",
            (root / "docs/DEMO_VIDEO_SCRIPT_BILINGUAL.md").exists(),
            "shot list, narration and subtitles",
        ),
        _check(
            "deployment documentation",
            (root / "docs/DATABASE_AND_DOCKER.md").exists(),
            "operations and recovery runbook",
        ),
    ]
    video_dir = root / "release/video"
    videos = (
        [p for p in video_dir.glob("*.mp4") if p.stat().st_size > 1024]
        if video_dir.exists()
        else []
    )
    physical = (
        _json_status(field_qc, "complete")
        and _json_status(live_mic, "passed")
        and _load(live_mic).get("hardware_validated") is True
        and _load(live_mic).get("mode") == "live_microphone"
    )
    manual = [
        _check("room/device recordings", _json_status(field_qc, "complete"), str(field_qc), True),
        _check("live microphone acceptance", physical, str(live_mic), True),
        _check("demo video recording", bool(videos), str(video_dir), True),
    ]
    software = software_checks(root)
    required = (
        software if mode == "software-final" else manual if mode in {"final", "field-final"} else []
    )
    relevant = [*non_field, *required]
    status = "passed" if all(row["passed"] for row in relevant) else "blocked"
    artifact_paths = [
        root / "artifacts/l4-engineering/crnn_short_event_calibrated.pt",
        root / "artifacts/l4-engineering/crnn_short_event_fp32.onnx",
        root / "reports/generated/l4-engineering/l4_summary.json",
        root / "data/l4-field-kit/reference_playback.wav",
        root / "reports/generated/software-release/robustness.json",
        root / "reports/generated/software-release/stream_soak.json",
        root / "reports/generated/software-release/video_validation.json",
        root / "release/video/AudioScope_v1.0_zh-CN.mp4",
        root / "release/video/zh-CN.srt",
        root / "release/video/en.srt",
    ]
    checksums = {
        path.relative_to(root).as_posix(): _sha256(path) for path in artifact_paths if path.exists()
    }
    return {
        "mode": mode,
        "status": status,
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "non_field_complete": all(row["passed"] for row in non_field),
        "release_name": "AudioScope v1.0 Software-Validated Release",
        "software_release_complete": all(row["passed"] for row in [*non_field, *software]),
        "field_release_complete": all(row["passed"] for row in [*non_field, *manual]),
        "physical_validation_performed": physical,
        "final_release_complete": all(row["passed"] for row in [*non_field, *manual]),
        "checks": [
            {**row, "required": row in relevant} for row in [*non_field, *software, *manual]
        ],
        "remaining_manual_actions": [row["name"] for row in manual if not row["passed"]],
        "artifact_sha256": checksums,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AudioScope release readiness")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--mode",
        choices=("preflight", "final", "software-final", "field-final"),
        default="preflight",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root, args.mode)
    output = args.output or args.root / "reports/generated/release" / f"{args.mode}_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    checksum_path = output.parent / "artifact_checksums.sha256"
    checksum_path.write_text(
        "".join(f"{digest}  {path}\n" for path, digest in result["artifact_sha256"].items()),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
