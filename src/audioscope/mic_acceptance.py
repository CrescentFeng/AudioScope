from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .engineering import stream_file, stream_microphone


def _duplicates(events: list[dict[str, Any]]) -> int:
    keys = [(row["label"], row["onset_s"], row["offset_s"]) for row in events]
    return len(keys) - len(set(keys))


def simulate(
    audio: Path,
    model: Path,
    onnx: Path | None,
    chunk_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = stream_file(
        audio.resolve(),
        model.resolve(),
        chunk_seconds=chunk_seconds,
        onnx_path=onnx.resolve() if onnx else None,
    )
    wall_seconds = time.perf_counter() - started
    p95 = float(result["compute_latency_ms"]["p95"])
    checks = {
        "processed_full_file": abs(result["processed_seconds"] - 119.361125) < 0.1
        if audio.name == "l4_long_demo.wav"
        else result["processed_seconds"] > 0,
        "compute_p95_below_chunk": p95 < chunk_seconds * 1000,
        "no_duplicate_events": _duplicates(result["events"]) == 0,
        "realtime_factor_below_one": wall_seconds / max(result["processed_seconds"], 1e-9) < 1.0,
    }
    return {
        "mode": "file_simulation",
        "hardware_validated": False,
        "status": "passed" if all(checks.values()) else "failed",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source": str(audio.resolve()),
        "backend": result["backend"],
        "chunk_seconds": chunk_seconds,
        "processed_seconds": result["processed_seconds"],
        "wall_seconds": wall_seconds,
        "realtime_factor": wall_seconds / max(result["processed_seconds"], 1e-9),
        "compute_latency_ms": result["compute_latency_ms"],
        "event_count": len(result["events"]),
        "duplicate_event_count": _duplicates(result["events"]),
        "checks": checks,
        "limitation": "文件流验证不能替代声卡权限、输入溢出和真实麦克风链路验收。",
    }


def live(
    model: Path,
    onnx: Path | None,
    duration_seconds: float,
    chunk_seconds: float,
) -> dict[str, Any]:
    try:
        import sounddevice as sd

        device = sd.query_devices(kind="input")
        started = time.perf_counter()
        result = stream_microphone(
            model.resolve(),
            duration_seconds=duration_seconds,
            chunk_seconds=chunk_seconds,
            onnx_path=onnx.resolve() if onnx else None,
        )
        elapsed = time.perf_counter() - started
        checks = {
            "captured_requested_duration": result["recorded_seconds"] >= duration_seconds * 0.99,
            "wall_time_near_realtime": elapsed <= duration_seconds * 1.25 + 2.0,
            "no_duplicate_events": _duplicates(result["events"]) == 0,
        }
        return {
            "mode": "live_microphone",
            "hardware_validated": True,
            "status": "passed" if all(checks.values()) else "failed",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "device": {
                "name": device.get("name"),
                "max_input_channels": device.get("max_input_channels"),
                "default_samplerate": device.get("default_samplerate"),
            },
            "duration_seconds": duration_seconds,
            "chunk_seconds": chunk_seconds,
            "event_count": len(result["events"]),
            "duplicate_event_count": _duplicates(result["events"]),
            "checks": checks,
            "result": result,
        }
    except Exception as error:
        return {
            "mode": "live_microphone",
            "hardware_validated": False,
            "status": "failed",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="AudioScope streaming acceptance recorder")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--simulate-file", type=Path)
    parser.add_argument("--live-seconds", type=float)
    parser.add_argument("--chunk-seconds", type=float, default=1.0)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/generated/field/mic_acceptance.json")
    )
    args = parser.parse_args()
    if bool(args.simulate_file) == bool(args.live_seconds):
        parser.error("choose exactly one of --simulate-file or --live-seconds")
    if args.simulate_file:
        result = simulate(args.simulate_file, args.model, args.onnx, args.chunk_seconds)
    else:
        result = live(args.model, args.onnx, args.live_seconds, args.chunk_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
