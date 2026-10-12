"""Audit, catalog and archive the workbench enhancement without modifying v1.0."""

import argparse
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import httpx
from sqlalchemy import func, select

from audioscope.catalog_import import _stable_id
from audioscope.db import AudioFile, BenchmarkResult, ModelArtifact, ProcessingRun, ReleaseAudit
from audioscope.model_registry import REGISTRY, resolve_model
from audioscope.release_catalog import sessions
from audioscope.software_validation import write_json
from audioscope.workbench import WorkbenchStore, sha256

OUT = Path("reports/generated/workbench-release")
IMPROVEMENT = Path("reports/generated/model-improvement")
WORKBENCH = Path("reports/generated/workbench")
FIXTURE = "b76d340c-cf75-4fa7-b1e1-9fdfa941b600"


def read(root, path):
    return json.loads((root / path).read_text())


def command(args):
    return subprocess.check_output(args, text=True).strip()


def audit(root):
    checks = []

    def check(name, value, evidence):
        checks.append({"name": name, "passed": bool(value), "evidence": evidence})
        if not value:
            raise RuntimeError(f"Release check failed: {name}")

    quality = read(root, IMPROVEMENT / "heldout_summary.json")
    development = read(root, IMPROVEMENT / "development_summary.json")
    selection = read(root, IMPROVEMENT / "decoder_selection.json")
    soak = read(root, IMPROVEMENT / "stream_soak.json")
    retrieval = read(root, WORKBENCH / "retrieval_evaluation.json")
    deployed = read(root, WORKBENCH / "container-verification.json")
    environment = read(root, WORKBENCH / "container-environment.json")
    registry = read(root, REGISTRY)
    model, onnx = resolve_model(root)
    check(
        "validation_only_selection",
        not development["test_used"] and not selection["test_used"],
        [
            str(IMPROVEMENT / "development_summary.json"),
            str(IMPROVEMENT / "decoder_selection.json"),
        ],
    )
    for variant in ("paired_baseline", "balanced_domain_augmented"):
        for seed in (2027, 2028, 2029):
            path = IMPROVEMENT / f"{variant}-{seed}.json"
            trial = read(root, path)
            checkpoint = root / "artifacts/model-improvement" / f"{variant}-{seed}.pt"
            check(
                f"trial_{variant}_{seed}",
                trial["seed"] == seed and sha256(checkpoint) == trial["sha256"],
                str(path),
            )
    expected_models = set(quality["protocol"]["models"])
    expected_conditions = set(quality["protocol"]["conditions"])
    expected_pairs = {
        (model, condition) for model in expected_models for condition in expected_conditions
    }
    check(
        "complete_heldout_grid",
        len(expected_models) == 8
        and len(expected_conditions) == 7
        and quality["protocol"]["clips"] == 221
        and {(row["model"], row["condition"]) for row in quality["results"]} == expected_pairs,
        str(IMPROVEMENT / "heldout_summary.json"),
    )
    check(
        "quality_gate",
        quality["quality_gate_passed"] and all(quality["promotion_checks"].values()),
        quality["promotion_checks"],
    )
    check(
        "registered_evidence_hashes",
        registry["quality_report_sha256"] == sha256(root / IMPROVEMENT / "heldout_summary.json")
        and registry["stream_report_sha256"] == sha256(root / IMPROVEMENT / "stream_soak.json"),
        str(REGISTRY),
    )
    check(
        "bounded_stream_1800",
        soak["status"] == "passed"
        and soak["processed_seconds"] >= 1800
        and all(soak["checks"].values())
        and soak["model_sha256"] == sha256(model)
        and soak["onnx_sha256"] == sha256(onnx),
        str(IMPROVEMENT / "stream_soak.json"),
    )
    check(
        "retrieval_evaluated",
        retrieval["status"] == "completed"
        and retrieval["protocol"]["clips"] == 221
        and retrieval["evaluated_queries"] == 120
        and not retrieval["protocol"]["tuning"]
        and all(math.isfinite(v) for v in retrieval["aggregate"].values()),
        str(WORKBENCH / "retrieval_evaluation.json"),
    )
    check(
        "container_workflow",
        deployed["status"] == "passed"
        and len(deployed["checks"]) == 24
        and all(deployed["checks"].values())
        and deployed["model_sha256"] == sha256(model),
        str(WORKBENCH / "container-verification.json"),
    )
    local_source = {
        str(p.relative_to(root / "src/audioscope")): sha256(p)
        for p in (root / "src/audioscope").rglob("*.py")
    }
    check(
        "installed_source",
        environment["status"] == "passed" and environment["runtime"]["source"] == local_source,
        str(WORKBENCH / "container-environment.json"),
    )
    check(
        "fresh_repeat_migration",
        environment["fresh_migration"]["second_upgrade_succeeded"]
        and environment["fresh_migration"]["revision"] == "0002_workbench"
        and environment["pip_check"] == "No broken requirements found.",
        str(WORKBENCH / "container-environment.json"),
    )
    image_id = command(
        ["docker", "image", "inspect", "audioscope-python:local", "--format", "{{.Id}}"]
    )
    check(
        "same_audited_image", image_id == deployed["image_id"] == environment["image_id"], image_id
    )
    for service in ("api", "demo"):
        container_id = command(["docker", "compose", "ps", "-q", service])
        running = json.loads(command(["docker", "inspect", container_id]))[0]
        check(
            f"running_{service}_image",
            running["State"]["Running"] and running["Image"] == image_id,
            container_id,
        )
    with httpx.Client(trust_env=False, timeout=15) as client:
        health = client.get("http://127.0.0.1:8002/health").raise_for_status().json()
        page = client.get("http://127.0.0.1:7862/config").raise_for_status().json()
    check(
        "live_services",
        health["revision"] == "0002_workbench" and "AudioScope" in page["title"],
        health,
    )
    junit = ElementTree.parse(root / WORKBENCH / "tests.xml").getroot()
    suites = list(junit.iter("testsuite"))
    check(
        "regression_tests",
        sum(int(s.attrib["tests"]) for s in suites) >= 75
        and all(
            int(s.attrib.get(k, 0)) == 0 for s in suites for k in ("failures", "errors", "skipped")
        ),
        str(WORKBENCH / "tests.xml"),
    )
    check(
        "ruff",
        "All checks passed!"
        in command([str(root / ".venv/bin/ruff"), "check", "src", "tests", "scripts"]),
        "ruff check src tests scripts",
    )
    store = WorkbenchStore(root / "runtime/workbench", sessions(root).kw["bind"].url)
    fixture = store.get(FIXTURE)
    check(
        "browser_review_persisted",
        fixture["revision"] >= 2
        and fixture["payload"]["events"][0]["offset_s"] == 2.6
        and fixture["review"]["events"][0]["offset_s"] == 2.5
        and fixture["review"]["source"] == "software_verification_fixture"
        and fixture["review"]["training_eligible"] is False,
        FIXTURE,
    )
    visual = read(root, WORKBENCH / "browser-verification.json")
    check(
        "observed_browser_flows",
        visual["status"] == "passed"
        and visual["final_image_id"] == image_id
        and all(visual["checks"].values())
        and all(sha256(root / p) == digest for p, digest in visual["screenshots"].items()),
        str(WORKBENCH / "browser-verification.json"),
    )
    check(
        "original_model_preserved",
        sha256(root / "artifacts/l4-engineering/crnn_short_event_calibrated.pt")
        == "297c5ed4a6db7e0944117a9ffa5e986fac3f503095baffa834e7a997ee19aff0",
        "v1.0 original checkpoint",
    )
    evidence = {}
    for directory in (
        IMPROVEMENT,
        WORKBENCH,
        Path("docs"),
        Path("src"),
        Path("tests"),
        Path("scripts"),
        Path("configs"),
        Path("migrations"),
    ):
        for path in sorted((root / directory).rglob("*")):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix in {".py", ".json", ".md", ".xml", ".yaml", ".jpg"}
            ):
                evidence[str(path.relative_to(root))] = sha256(path)
    for name in (
        "README.md",
        "README.en.md",
        "Dockerfile",
        "compose.yaml",
        "pyproject.toml",
        "alembic.ini",
        str(REGISTRY),
    ):
        evidence[name] = sha256(root / name)
    report = {
        "status": "passed",
        "release": "AudioScope Workbench Enhancement 2026-10-11",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "software_release_complete": True,
        "field_release_complete": False,
        "physical_validation_performed": False,
        "scope": "Four priority improvements; review workbench (A); local text retrieval (B)",
        "archive_verification": "separate archive_verification.json",
        "image_id": image_id,
        "checks": checks,
        "evidence_sha256": evidence,
    }
    write_json(root / OUT / "final_audit.json", report)
    return report


def sync(root, audit_report):
    paths = [
        IMPROVEMENT / "heldout_summary.json",
        IMPROVEMENT / "stream_soak.json",
        WORKBENCH / "retrieval_evaluation.json",
        WORKBENCH / "container-verification.json",
    ]
    ids, metric_ids = [], []
    with sessions(root)() as session:
        audio_count = session.scalar(select(func.count()).select_from(AudioFile))
        for path in paths:
            report, digest = read(root, path), sha256(root / path)
            run_id = _stable_id("workbench-enhancement", str(path), digest)
            ids.append(run_id)
            session.merge(
                ProcessingRun(
                    id=run_id,
                    run_kind=f"workbench_{path.stem}",
                    status=report["status"],
                    finished_at=datetime.now(timezone.utc),
                    environment_json={"physical_validation_performed": False},
                    summary_json={
                        "report_uri": f"audioscope://{path}",
                        "sha256": digest,
                        "image_id": audit_report["image_id"],
                    },
                )
            )
            session.flush()
            rows = []
            if path.stem == "heldout_summary":
                for row in report["results"]:
                    metrics = row["metrics"]
                    for name, value in {
                        "event_f1": metrics["event_based"]["micro"]["f1"],
                        "macro_event_f1": metrics["event_based"]["macro_f1"],
                        "precision": metrics["event_based"]["micro"]["precision"],
                        "recall": metrics["event_based"]["micro"]["recall"],
                        "short_event_f1": row["duration_metrics"]["short"]["micro"]["f1"],
                    }.items():
                        rows.append((row["model"] + "/" + row["condition"], name, value))
            elif path.stem == "stream_soak":
                rows = [
                    ("decoder_candidate", name, report[name])
                    for name in (
                        "processed_seconds",
                        "rtf",
                        "compute_p95_ms",
                        "rss_growth_bytes_after_warmup",
                    )
                ]
            elif path.stem == "retrieval_evaluation":
                rows = [
                    ("positive_queries_only", name, value)
                    for name, value in report["aggregate"].items()
                ]
            for slice_name, name, value in rows:
                if not math.isfinite(value):
                    raise ValueError("Nonfinite catalog metric")
                metric_id = _stable_id(run_id, slice_name, name)
                metric_ids.append(metric_id)
                session.merge(
                    BenchmarkResult(
                        id=metric_id,
                        run_id=run_id,
                        metric_name=name,
                        metric_value=value,
                        slice_name=slice_name,
                        metadata_json={"software_validation": True, "source_sha256": digest},
                    )
                )
        model, _ = resolve_model(root)
        relative, digest = model.relative_to(root).as_posix(), sha256(model)
        session.merge(
            ModelArtifact(
                id=_stable_id(relative, digest),
                run_id=ids[0],
                name=model.name,
                stage="workbench-enhancement",
                format="pt",
                storage_uri=f"audioscope-artifact://{relative.removeprefix('artifacts/')}",
                sha256=digest,
                byte_size=model.stat().st_size,
                metadata_json={"neural_weights_unchanged": True},
            )
        )
        audit_id = _stable_id("workbench-final", sha256(root / OUT / "final_audit.json"))
        session.merge(
            ReleaseAudit(
                id=audit_id,
                mode="workbench-final",
                status="passed",
                report_uri=f"audioscope://{OUT}/final_audit.json",
                checks_json=audit_report["checks"],
            )
        )
        session.commit()
        counts = {
            "processing_runs": session.scalar(
                select(func.count()).select_from(ProcessingRun).where(ProcessingRun.id.in_(ids))
            ),
            "benchmark_results": session.scalar(
                select(func.count())
                .select_from(BenchmarkResult)
                .where(BenchmarkResult.id.in_(metric_ids))
            ),
            "audio_files": session.scalar(select(func.count()).select_from(AudioFile)),
        }
        if counts != {
            "processing_runs": len(ids),
            "benchmark_results": len(metric_ids),
            "audio_files": audio_count,
        }:
            raise RuntimeError("Catalog read-back mismatch")
    return {
        "status": "passed",
        **counts,
        "audit_id": audit_id,
        "run_ids": ids,
        "source_sha256": {str(p): sha256(root / p) for p in paths},
    }


def archive(root, destination):
    destination.mkdir(parents=True, exist_ok=False)
    folders = {
        "src": "01_源码/src",
        "scripts": "01_源码/scripts",
        "tests": "02_测试",
        "configs": "03_配置/configs",
        "migrations": "03_配置/migrations",
        "docs": "04_文档",
        str(IMPROVEMENT): "05_识别与鲁棒性报告",
        str(WORKBENCH): "06_工作台与检索验收",
        str(OUT): "07_发布审计",
        "artifacts/model-improvement": "08_模型/model-improvement",
        "artifacts/workbench-release": "08_模型/workbench-release",
    }
    for source, target in folders.items():
        shutil.copytree(
            root / source,
            destination / target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        for path in (root / source).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                copied = destination / target / path.relative_to(root / source)
                if sha256(path) != sha256(copied):
                    raise RuntimeError("Archive source/copy checksum mismatch")
    for name in (
        "README.md",
        "README.en.md",
        "Dockerfile",
        "compose.yaml",
        "pyproject.toml",
        "alembic.ini",
        "LICENSE",
        "Makefile",
        "l4_app.py",
        "app.py",
        ".dockerignore",
        ".gitignore",
    ):
        shutil.copy2(root / name, destination / name)
    shutil.copy2(
        root / "reports/WORKBENCH_IMPROVEMENT.md", destination / "04_文档/WORKBENCH_IMPROVEMENT.md"
    )
    for name in ("crnn_short_event_calibrated.pt", "crnn_short_event_fp32.onnx"):
        shutil.copy2(root / "artifacts/l4-engineering" / name, destination / "08_模型" / name)
    (destination / "03_配置/data").mkdir()
    shutil.copy2(root / "data/sources.yaml", destination / "03_配置/data/sources.yaml")
    shutil.copy2(root / ".github/workflows/ci.yml", destination / "02_测试/ci.yml")
    (destination / "归档说明.txt").write_text(
        "这是按阶段分类的发布证据归档，不是可直接启动的独立安装包。\n"
        "运行入口、限制、恢复说明见 04_文档/WORKBENCH_GUIDE.md 与 WORKBENCH_RELEASE.md。\n"
        "重建项目时将 01_源码 内的 src/scripts、03_配置 内的 configs/migrations 放回项目根目录，"
        "将 08_模型 内模型恢复到登记的 artifacts 相对路径；还需已有数据与本地 CLAP 权重。\n"
        "出于隐私和体积控制，不含 .env、数据库凭据、用户音频、原始数据集和大模型权重。\n"
        "原 12_软件验证封板 不改动；本版本仍未进行自有物理设备/房间认证。\n",
        encoding="utf-8",
    )
    files = sorted(p for p in destination.rglob("*") if p.is_file())
    hashes = {str(p.relative_to(destination)): sha256(p) for p in files}
    (destination / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items())
    )
    if any(sha256(destination / name) != digest for name, digest in hashes.items()):
        raise RuntimeError("Archive verification failed")
    report = {
        "status": "passed",
        "destination": str(destination),
        "files_verified": len(files),
        "manifest_sha256": sha256(destination / "SHA256SUMS.txt"),
        "secrets_and_raw_audio_excluded": True,
        "source_audit_sha256": sha256(root / OUT / "final_audit.json"),
    }
    write_json(root / OUT / "archive_verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    report = audit(root)
    first = sync(root, report)
    second = sync(root, report)
    if first != second:
        raise RuntimeError("Catalog sync is not idempotent")
    write_json(root / OUT / "catalog_sync.json", {**second, "idempotence_verified": True})
    print(json.dumps(archive(root, args.destination.resolve()), ensure_ascii=False, indent=2))
