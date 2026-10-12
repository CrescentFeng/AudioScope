from __future__ import annotations

import hashlib
import shutil
from datetime import date
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
DESKTOP_ARCHIVE = Path.home() / "Desktop" / "AudioScope"
SNAPSHOT = DESKTOP_ARCHIVE / "L4工程阶段" / "19_当前软件项目快照"
OLD_MANUAL = DESKTOP_ARCHIVE / "AudioScope 使用手册.pdf"
MANUAL_HISTORY = DESKTOP_ARCHIVE / "历史版本" / f"AudioScope 使用手册_更新前_{date.today():%Y%m%d}.pdf"
NEW_MANUAL = SOURCE / "output" / "pdf" / "AudioScope 使用手册.pdf"

ROOT_FILES = [
    ".dockerignore", ".env.example", ".gitignore", "Dockerfile",
    "Dockerfile.offline-api", "Dockerfile.readiness", "Dockerfile.ui",
    "LICENSE", "Makefile", "README.md", "README.en.md", "alembic.ini",
    "app.py", "compose.yaml", "l4_app.py", "pyproject.toml",
]
DIRECTORIES = ["configs", "docs", "infra", "migrations", "scripts", "src", "tests", "artifacts", "reports"]
DATA_PATHS = [
    "data/README.md",
    "data/PRETRAINED_MODELS.md",
    "data/sources.yaml",
    "data/manifests",
    "data/smoke",
    "data/cnn-smoke",
    "data/sed-esc10-polyphonic",
    "data/l4-engineering/l4_long_demo.wav",
    "data/l4-field-kit",
    "data/external-real-pilot-v1",
    "data/external-strong-source-audit",
    "data/pretrained/clap-htsat-unfused-8fa0f1c",
]


def copy_path(relative: str, destination: Path) -> None:
    source = SOURCE / relative
    target = destination / relative
    if source.is_dir():
        shutil.copytree(source, target, ignore=shutil.ignore_patterns(".DS_Store", "__pycache__", ".pytest_cache"))
    elif source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    else:
        raise FileNotFoundError(f"Required project path missing: {source}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not DESKTOP_ARCHIVE.is_dir():
        raise FileNotFoundError(f"Desktop project archive not found: {DESKTOP_ARCHIVE}")
    if SNAPSHOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing snapshot: {SNAPSHOT}")
    if not NEW_MANUAL.is_file() or not OLD_MANUAL.is_file():
        raise FileNotFoundError("Both current and existing desktop manuals must be present.")
    if MANUAL_HISTORY.exists():
        raise FileExistsError(f"Refusing to overwrite historical manual: {MANUAL_HISTORY}")

    SNAPSHOT.mkdir(parents=True)
    for relative in ROOT_FILES + DIRECTORIES + DATA_PATHS:
        copy_path(relative, SNAPSHOT)

    # Include release deliverables, but omit the video builder's temporary chunks/logs.
    copy_path("release/video/AudioScope_v1.0_zh-CN.mp4", SNAPSHOT)
    copy_path("release/video/build_manifest.json", SNAPSHOT)
    copy_path("release/video/zh-CN.srt", SNAPSHOT)
    copy_path("release/video/en.srt", SNAPSHOT)
    (SNAPSHOT / "docs" / "AudioScope 使用手册.pdf").parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(NEW_MANUAL, SNAPSHOT / "docs" / "AudioScope 使用手册.pdf")

    guide = """AudioScope 当前软件项目快照\n\n""" \
        "本目录是截至 2026-10-12 的源码与运行资产快照。它包含最新源码、配置、迁移、测试、模型、演示素材、审核报告、外部录音证据和发布交付物。\n\n" \
        "桌面根目录的“打开AudioScope.command”是已配置的启动入口，服务实际运行在 Documents/ChatGPT/Audio Programs/01-audioscope 主项目目录；本快照用于资料归档与核验，不替代该主工作副本。\n\n" \
        "为避免泄露或复制无关私有/可再生数据，本快照不含 .env、.venv、数据库运行卷、历史工作台数据库、原始 DESED 全量训练语料或中间特征缓存。需要训练或完整复现实验时，请使用主项目数据目录和环境。`.env.example` 是公开配置模板。\n\n" \
        "发布结论仍为：AudioScope v1.0 软件验证完成；尚未进行自有物理设备/真实房间认证。外部素材和人工核对记录仅表示该批录音的验证结果，不构成设备认证。\n"
    (SNAPSHOT / "00_先读这里.txt").write_text(guide, encoding="utf-8")

    file_list = sorted(p for p in SNAPSHOT.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    with (SNAPSHOT / "SHA256SUMS.txt").open("w", encoding="utf-8") as stream:
        for path in file_list:
            stream.write(f"{sha256(path)}  {path.relative_to(SNAPSHOT).as_posix()}\n")

    # Keep the previous desktop manual recoverable before installing the latest one.
    MANUAL_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OLD_MANUAL, MANUAL_HISTORY)
    shutil.copy2(NEW_MANUAL, OLD_MANUAL)

    index = DESKTOP_ARCHIVE / "README_最新项目资料索引.txt"
    index.write_text(
        "AudioScope 桌面项目资料索引（更新于 2026-10-12）\n\n"
        "最新源码与运行资产快照：L4工程阶段/19_当前软件项目快照/\n"
        "最新详细使用手册：AudioScope 使用手册.pdf\n"
        "启动入口：打开AudioScope.command（连接到 Documents 中的主项目工作副本）\n\n"
        "保留：L1–L4 各阶段归档、外部录音及标注审核记录、现场采集工具包、旧版发布审计。它们分别是阶段证据、可追溯数据或未来现场扩展材料，不是当前运行副本的冗余。\n\n"
        "不复制到当前快照：.env（含本机秘密）、.venv（可重建环境）、运行数据库和工作台历史、全量训练集及特征缓存、视频构建临时文件。它们仍保留在 Documents 主项目中；快照内含 .env.example、推理需要的模型与样例。\n\n"
        "可安全清理：AudioScope 目录中的 .DS_Store 均为 Finder 自动生成的显示状态文件，删除不影响项目，Finder 以后可能重新生成。未清理任何阶段归档、模型、报告、录音或标注文件。\n\n"
        "准确性说明：桌面快照是当前完整软件资料归档，不是脱离主项目即可启动的独立安装包；软件验证不等于自有物理设备/房间认证。\n",
        encoding="utf-8",
    )

    removed_metadata = 0
    for path in DESKTOP_ARCHIVE.rglob(".DS_Store"):
        if path.is_file():
            path.unlink()
            removed_metadata += 1

    print(f"snapshot={SNAPSHOT}")
    print(f"snapshot_files={len(file_list) + 2}")
    print(f"snapshot_bytes={sum(p.stat().st_size for p in SNAPSHOT.rglob('*') if p.is_file())}")
    print(f"snapshot_manual_sha256={sha256(SNAPSHOT / 'docs' / 'AudioScope 使用手册.pdf')}")
    print(f"desktop_manual_sha256={sha256(OLD_MANUAL)}")
    print(f"preserved_old_manual={MANUAL_HISTORY}")
    print(f"removed_finder_metadata={removed_metadata}")


if __name__ == "__main__":
    main()
