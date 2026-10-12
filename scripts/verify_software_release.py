"""Save actual lint/test exit codes and output as release evidence."""

import json
import subprocess
import sys
from pathlib import Path

from audioscope.release_audit import _sha256

root = Path(__file__).resolve().parents[1]
commands = [
    [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts", "app.py", "l4_app.py"],
    [sys.executable, "-m", "pytest"],
]
results = []
for command in commands:
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    results.append(
        {
            "command": command,
            "exit_code": result.returncode,
            "output": result.stdout + result.stderr,
        }
    )
report = {
    "status": "passed" if all(r["exit_code"] == 0 for r in results) else "failed",
    "results": results,
    "source_sha256": {
        str(p.relative_to(root)): _sha256(p)
        for folder in (root / "src/audioscope", root / "tests")
        for p in sorted(folder.glob("*.py"))
    },
}
path = root / "reports/generated/software-release/code_verification.json"
path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
for result in results:
    print(result["output"])
raise SystemExit(0 if report["status"] == "passed" else 1)
