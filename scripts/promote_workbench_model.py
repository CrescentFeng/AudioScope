"""Register the validation-selected decoder only after fixed quality and stream gates."""

import json
from pathlib import Path

from audioscope.model_registry import REGISTRY
from audioscope.software_validation import MODEL, ONNX, write_json
from audioscope.workbench import sha256


def main():
    root = Path.cwd()
    report_root = root / "reports/generated/model-improvement"
    heldout = json.loads((report_root / "heldout_summary.json").read_text())
    selection = json.loads((report_root / "decoder_selection.json").read_text())
    soak = json.loads((report_root / "stream_soak.json").read_text())
    assert heldout["quality_gate_passed"] and all(heldout["promotion_checks"].values())
    assert soak["status"] == "passed" and soak["processed_seconds"] >= 1800
    candidate = Path(selection["candidate"]).resolve()
    assert candidate.is_relative_to(root / "artifacts")
    assert sha256(candidate) == selection["candidate_sha256"] == soak["model_sha256"]
    assert sha256(root / ONNX) == soak["onnx_sha256"]
    models = {}
    for key, label, path in [
        ("v1_baseline", "v1.0 基线", root / MODEL),
        ("v1_refined", "事件边界优化版（默认）", candidate),
    ]:
        models[key] = {
            "label": label,
            "checkpoint": str(path.relative_to(root)),
            "onnx": str(ONNX),
            "checkpoint_sha256": sha256(path),
            "onnx_sha256": sha256(root / ONNX),
        }
    write_json(
        root / REGISTRY,
        {
            "default": "v1_refined",
            "models": models,
            "quality_report_sha256": sha256(report_root / "heldout_summary.json"),
            "stream_report_sha256": sha256(report_root / "stream_soak.json"),
            "v1_archive_modified": False,
        },
    )
    print("Workbench default registered: v1_refined; original v1.0 artifacts unchanged.")


if __name__ == "__main__":
    main()
