from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def get_json(base_url: str, path: str):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"{base_url.rstrip('/')}{path}", timeout=10) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/generated/docker/validation.json")
    )
    args = parser.parse_args()
    health = get_json(args.base_url, "/health")
    readiness = get_json(args.base_url, "/v1/readiness")
    datasets = get_json(args.base_url, "/v1/datasets")
    checks = {
        "database_ready": health.get("database") == "ready",
        "migration_current": health.get("revision") == "0001_audioscope_catalog",
        "catalog_ready": readiness.get("status") == "ready",
        "dataset_count": len(datasets) == 2,
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "service": "api-offline compatibility container",
        "checks": checks,
        "health": health,
        "readiness": readiness,
        "datasets": datasets,
        "note": (
            "FastAPI remains the primary API; this cached-image adapter verifies Docker "
            "networking and database queries when external registries are unavailable."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
