"""Reliable loopback-only default launcher; never takes over an occupied port."""

from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path


def bypass_loopback_proxy():
    for key in ("NO_PROXY", "no_proxy"):
        values = [v.strip() for v in os.environ.get(key, "").split(",") if v.strip()]
        os.environ[key] = ",".join(dict.fromkeys([*values, "127.0.0.1", "localhost", "::1"]))


def available_port(start=7861, count=10):
    for port in range(start, start + count):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"{start}–{start + count - 1} 端口均不可用，请用 --port 指定其他端口。")


def main():
    parser = argparse.ArgumentParser(description="启动 AudioScope 声音事件工作台")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--server-name",
        default="127.0.0.1",
        choices=["127.0.0.1", "0.0.0.0"],
        help="容器内部可使用 0.0.0.0；宿主机发布端口仍应绑定本机",
    )
    parser.add_argument(
        "--local-history",
        action="store_true",
        help="明确使用本地 SQLite 历史，不连接已配置的 PostgreSQL",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    bypass_loopback_proxy()
    if args.local_history:
        os.environ["AUDIOSCOPE_LOCAL_HISTORY"] = "1"
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    from .model_registry import resolve_model

    try:
        model, onnx = resolve_model(root)
    except ValueError as exc:
        parser.error(str(exc))
    port = available_port(args.port or 7861, 1 if args.port else 10)
    # Import lazily after process-local proxy setup.
    from .workbench_ui import build_workbench

    print(f"AudioScope 工作台：http://127.0.0.1:{port}（仅本机；Ctrl+C 停止）", flush=True)
    build_workbench(model, onnx, root).launch(
        server_name=args.server_name,
        server_port=port,
        share=False,
        max_file_size="100mb",
    )


if __name__ == "__main__":
    main()
