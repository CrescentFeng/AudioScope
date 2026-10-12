"""Local readiness check. No writes, training, user audio, or secret-bearing errors."""

import argparse
import json
import urllib.request
from pathlib import Path


def page_ready(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url.rstrip("/") + "/config", timeout=4) as response:
        config = json.load(response)
    return (
        str(config.get("title", "")).startswith("AudioScope")
        and any(
            c.get("props", {}).get("elem_id") == "scope-header"
            for c in config.get("components", [])
        )
        and any(d.get("api_name") == "analyse" for d in config.get("dependencies", []))
    )


def check(root, url, inference=False):
    from sqlalchemy import create_engine, text

    from .workbench_ui import history_database_url

    if not page_ready(url):
        raise RuntimeError("页面标识或分析接口不正确")
    database = history_database_url(root)
    if database is None:
        raise RuntimeError("正式入口必须使用已配置的 PostgreSQL，不自动切换空白本地历史")
    engine = create_engine(database, connect_args={"connect_timeout": 4}, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SET statement_timeout = '4000ms'"))
            conn.execute(text("SELECT 1 FROM workbench_jobs LIMIT 0"))
    finally:
        engine.dispose()
    result = {"page": True, "database": True, "inference_tested": False}
    if inference:
        import numpy as np

        from .engineering import SEDDeployment
        from .model_registry import resolve_model

        model, onnx = resolve_model(root)
        deployment = SEDDeployment(model, onnx)
        prediction = deployment.predict_samples(
            np.zeros(deployment.sample_rate, dtype=np.float32), include_unknown=False
        )
        if not np.isfinite(prediction["probabilities"]).all():
            raise RuntimeError("模型返回非有限值")
        result.update(
            inference_tested=True, model_sha256=deployment.provenance["checkpoint_sha256"]
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/app"))
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--inference", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps({"status": "ready", **check(args.root, args.url, args.inference)}))
    except Exception as exc:
        # Database exceptions can contain connection data; do not print raw exception text.
        print(json.dumps({"status": "not_ready", "error_type": type(exc).__name__}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
