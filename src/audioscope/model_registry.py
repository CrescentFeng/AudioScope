"""Explicit deployed-model choice; never select an experiment based on test ranking."""

import json
from pathlib import Path

from .workbench import sha256

REGISTRY = Path("artifacts/workbench-release/model_registry.json")


def model_registry(root):
    root = Path(root).resolve()
    if not (root / REGISTRY).is_file():
        return {
            "default": "v1_baseline",
            "models": {
                "v1_baseline": {
                    "label": "v1.0 基线",
                    "checkpoint": "artifacts/l4-engineering/crnn_short_event_calibrated.pt",
                    "onnx": "artifacts/l4-engineering/crnn_short_event_fp32.onnx",
                }
            },
        }
    return json.loads((root / REGISTRY).read_text())


def resolve_model(root, key="current"):
    root = Path(root).resolve()
    registry = model_registry(root)
    key = registry["default"] if key == "current" else key
    if key not in registry["models"]:
        raise ValueError("模型未注册，请选择已验收的模型版本。")
    entry = registry["models"][key]
    resolved = []
    for field in ("checkpoint", "onnx"):
        path = (root / entry[field]).resolve()
        if not path.is_relative_to(root / "artifacts") or not path.is_file():
            raise ValueError("模型文件不在已配置的项目 artifacts 目录内。")
        if entry.get(f"{field}_sha256") and sha256(path) != entry[f"{field}_sha256"]:
            raise ValueError("模型哈希与发布登记不符，请先重新验收。")
        resolved.append(path)
    return tuple(resolved)
