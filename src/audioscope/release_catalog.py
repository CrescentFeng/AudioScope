"""Publish software evidence to the existing catalog, without storing media blobs."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import URL, func, select

from .catalog_import import _stable_id, verify_catalog
from .db import BenchmarkResult, ProcessingRun, ReleaseAudit, make_session_factory
from .release_audit import _sha256
from .software_validation import OUT, write_json


def sessions(root):
    url = os.getenv("AUDIOSCOPE_DATABASE_URL")
    if not url:
        # Read a trusted local configuration as data; never execute it or print secrets.
        values = dict(
            line.split("=", 1)
            for line in (root / ".env").read_text().splitlines()
            if line.strip() and not line.startswith("#") and "=" in line
        )
        url = URL.create(
            "postgresql+psycopg",
            username="audioscope",
            password=values["AUDIOSCOPE_DB_PASSWORD"],
            host="127.0.0.1",
            port=int(values.get("AUDIOSCOPE_DB_PORT", "55432")),
            database="audioscope",
        )
    return make_session_factory(url)


def snapshot(root):
    with sessions(root)() as session:
        verification = verify_catalog(session)
    result = {
        "status": verification["status"],
        "verification": verification,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "source": "live PostgreSQL query",
    }
    write_json(root / OUT / "catalog_snapshot.json", result)
    return result


def sync(root):
    paths = [OUT / f"{name}.json" for name in ("robustness", "stream_soak", "video_validation")]
    with sessions(root)() as session:
        run_ids = []
        for path in paths:
            report = json.loads((root / path).read_text())
            digest = _sha256(root / path)
            run_id = _stable_id("software-v1", str(path), digest)
            run_ids.append(run_id)
            session.merge(
                ProcessingRun(
                    id=run_id,
                    run_kind=f"software_{path.stem}",
                    status=report["status"],
                    finished_at=datetime.now(timezone.utc),
                    environment_json={"physical_validation_performed": False},
                    summary_json={
                        "report_uri": f"audioscope://{path}",
                        "sha256": digest,
                        "evidence": report,
                    },
                )
            )
            session.flush()
            rows = report.get("conditions", [report])
            for index, row in enumerate(rows):
                for metric in (
                    "event_f1",
                    "macro_event_f1",
                    "segment_f1",
                    "model_rtf",
                    "rtf",
                    "compute_p95_ms",
                    "rss_growth_bytes_after_warmup",
                    "duration_s",
                ):
                    if metric not in row:
                        continue
                    session.merge(
                        BenchmarkResult(
                            id=_stable_id(run_id, index, metric),
                            run_id=run_id,
                            metric_name=metric,
                            metric_value=float(row[metric]),
                            slice_name=row.get("condition", path.stem),
                            metadata_json={"software_validation": True},
                        )
                    )
        audit_path = root / "reports/generated/release/software-final_audit.json"
        audit_id = None
        if audit_path.exists():
            report = json.loads(audit_path.read_text())
            audit_id = _stable_id("software-final", _sha256(audit_path))
            session.merge(
                ReleaseAudit(
                    id=audit_id,
                    mode="software-final",
                    status=report["status"],
                    report_uri="audioscope://reports/generated/release/software-final_audit.json",
                    checks_json=report["checks"],
                )
            )
        session.commit()
        count = session.scalar(
            select(func.count()).select_from(ProcessingRun).where(ProcessingRun.id.in_(run_ids))
        )
        metrics = session.scalar(
            select(func.count())
            .select_from(BenchmarkResult)
            .where(BenchmarkResult.run_id.in_(run_ids))
        )
    result = {
        "status": "passed" if count == 3 and metrics >= 28 else "failed",
        "processing_runs": count,
        "benchmark_results": metrics,
        "audit_id": audit_id,
        "source_sha256": {str(p): _sha256(root / p) for p in paths},
    }
    write_json(root / OUT / "catalog_sync.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["snapshot", "sync"])
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = (snapshot if args.command == "snapshot" else sync)(args.root.resolve())
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
