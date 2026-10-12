"""Create the desktop release archive, excluding secrets and original dataset audio."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from audioscope.release_audit import _sha256, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if audit(root, "software-final")["status"] != "passed":
        raise SystemExit("Software release gate must pass before packaging")
    target = args.destination.resolve()
    target.mkdir(parents=True, exist_ok=False)
    trees = {
        "reports/generated/software-release": "01_评测报告",
        "reports/generated/release": "02_发布审计",
        "src/audioscope": "05_源码/audioscope",
        "tests": "06_回归测试",
    }
    for source, dest in trees.items():
        shutil.copytree(
            root / source, target / dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    video = target / "03_演示视频与字幕"
    video.mkdir()
    for name in ("AudioScope_v1.0_zh-CN.mp4", "zh-CN.srt", "en.srt", "build_manifest.json"):
        shutil.copy2(root / "release/video" / name, video / name)
    docs = target / "04_发布文档"
    docs.mkdir()
    for name in (
        "SOFTWARE_RELEASE.md",
        "L4_MODEL_CARD.md",
        "PROJECT_SUMMARY_BILINGUAL.md",
        "RESUME_BULLETS_BILINGUAL.md",
        "DEMO_VIDEO_SCRIPT_BILINGUAL.md",
    ):
        shutil.copy2(root / "docs" / name, docs / name)
    for name in ("pyproject.toml", "Makefile", "README.md", "README.en.md"):
        shutil.copy2(root / name, target / name)
    files = sorted(p for p in target.rglob("*") if p.is_file())
    entries = [(_sha256(p), p.relative_to(target)) for p in files]
    (target / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {path}\n" for digest, path in entries), encoding="utf-8"
    )
    if not all(_sha256(target / path) == digest for digest, path in entries):
        raise SystemExit("Archive hash verification failed")
    print(f"Archived and verified {len(files)} files at {target}")


if __name__ == "__main__":
    main()
