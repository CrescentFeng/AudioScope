"""Archive this UI-only iteration without replacing historical release snapshots."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/generated/simple-home"
DEST = Path.home() / "Desktop/AudioScope/L4工程阶段/17_简洁首页与按需展开"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    if DEST.exists():
        raise ValueError("Archive already exists; refusing to overwrite")
    registry = json.loads((ROOT / "artifacts/workbench-release/model_registry.json").read_text())
    for entry in registry["models"].values():
        for field in ("checkpoint", "onnx"):
            if sha(ROOT / entry[field]) != entry[field + "_sha256"]:
                raise ValueError("Model hash changed")
    shutil.copytree(OUT, DEST / "验收与截图")
    for relative in ("src/audioscope/workbench_ui.py", "src/audioscope/timeline.py",
                     "tests/test_workbench_ui.py", "tests/test_timeline.py", "Dockerfile.ui",
                     "scripts/archive_simple_home.py"):
        target = DEST / "改版代码" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    shutil.copy2(OUT / "界面改版说明.txt", DEST / "先读这里.txt")
    inventory = {str(p.relative_to(DEST)): sha(p) for p in sorted(DEST.rglob("*")) if p.is_file()}
    (DEST / "SHA256SUMS.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    if not all(sha(DEST / name) == value for name, value in inventory.items()):
        raise ValueError("Archive integrity check failed")
    print(json.dumps({"archive": str(DEST), "files_verified": len(inventory), "model_hashes_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
