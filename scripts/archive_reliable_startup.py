"""Archive verified startup changes and install the explicit desktop entry point."""

import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/generated/reliable-startup"
DESKTOP = Path.home() / "Desktop/AudioScope"
DEST = DESKTOP / "L4工程阶段/18_可靠启动与健康检查"
ENTRY = DESKTOP / "打开AudioScope.command"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    if DEST.exists() or ENTRY.exists():
        raise ValueError("Destination exists; refusing to overwrite")
    suites = list(ET.parse(OUT / "tests.xml").getroot().iter("testsuite"))
    if not suites or any(
        int(s.get(key, "0")) for s in suites for key in ("failures", "errors", "skipped")
    ):
        raise ValueError("Test evidence missing or not passing")
    report = json.loads((ROOT / "runtime/startup/latest.json").read_text())
    if report.get("status") != "ready" or not report.get("checks", {}).get("inference_tested"):
        raise ValueError("Successful live readiness evidence required")
    shutil.copytree(OUT, DEST / "验收")
    shutil.copy2(ROOT / "runtime/startup/latest.json", DEST / "验收/启动自检.json")
    for relative in (
        "scripts/open_audioscope.py",
        "scripts/archive_reliable_startup.py",
        "src/audioscope/workbench_health.py",
        "tests/test_startup.py",
        "compose.yaml",
        "Dockerfile.readiness",
        "打开AudioScope.command",
        "scripts/bootstrap_env.py",
        "docs/WORKBENCH_GUIDE.md",
    ):
        target = DEST / "实现" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    shutil.copy2(OUT / "可靠启动说明.txt", DEST / "先读这里.txt")
    shutil.copy2(ROOT / "打开AudioScope.command", ENTRY)
    ENTRY.chmod(0o755)
    inventory = {str(p.relative_to(DEST)): sha(p) for p in sorted(DEST.rglob("*")) if p.is_file()}
    (DEST / "SHA256SUMS.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n"
    )
    if not all(sha(DEST / name) == value for name, value in inventory.items()):
        raise ValueError("Archive integrity check failed")
    if sha(ENTRY) != sha(ROOT / "打开AudioScope.command"):
        raise ValueError("Desktop entry integrity check failed")
    print(
        json.dumps(
            {"archive": str(DEST), "entry": str(ENTRY), "files_verified": len(inventory)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
