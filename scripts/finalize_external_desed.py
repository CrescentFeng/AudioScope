"""Verify saved evidence, then optionally create a non-overwriting desktop archive."""
import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from validate_external_desed import DATA, EVENT_SETTINGS, OUT, ROOT, SOURCE, digest, write

from audioscope.sed_data import TimedEvent, read_soundscape_manifest
from audioscope.sed_metrics import evaluate_sed


def verify():
    report = json.loads((OUT / "evaluation.json").read_text())
    protocol = json.loads((OUT / "frozen-protocol.json").read_text())
    audit = json.loads((OUT / "data-audit.json").read_text())
    cases = json.loads((OUT / "event-comparisons.json").read_text())
    records = read_soundscape_manifest(DATA / "manifest.jsonl")
    estimates = {c["scene_id"]: [TimedEvent(e["label"], e["onset_s"], e["offset_s"], "prediction", "prediction")
                                for e in c["predictions"]] for c in cases}
    classes = list(report["metrics"]["event_based"]["per_class"])
    reconstructed = evaluate_sed(records, estimates, classes, EVENT_SETTINGS, 1.)
    old_audit = json.loads((ROOT / "reports/generated/workbench-release/final_audit.json").read_text())
    changed = [p for p, h in old_audit["evidence_sha256"].items() if digest(ROOT / p) != h]
    suites = ET.parse(OUT / "tests.xml").getroot().findall("testsuite")
    tests = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    retained_names = {Path(r.path).name for r in records}
    excluded_names = {r["filename"] for r in audit["source_id_exclusions"] + audit["additional_exclusions"]}
    references = [json.loads((DATA / "references" / f"{r.scene_id}.reference.json").read_text()) for r in records]
    checks = {
        "saved_predictions_reproduce_metrics": reconstructed == report["metrics"],
        "all_records_have_predictions": set(estimates) == {r.scene_id for r in records},
        "disjoint_inclusions_exclusions": not retained_names & excluded_names,
        "all_official_clips_accounted_for": len(retained_names) + len(excluded_names) == audit["official_recordings"],
        "final_references_exact_count": len(list((DATA / "references").glob("*.json"))) == len(records),
        "official_references_eligible": all(r["source"] == "dataset_reference" and r["formal_evaluation_eligible"] for r in references),
        "reference_audio_hashes_match": all(ref["audio_sha256"] == digest(DATA / r.path) for r, ref in zip(records, references)),
        "frozen_manifest_matches": digest(DATA / "manifest.jsonl") == protocol["manifest_sha256"],
        "frozen_data_audit_matches": digest(OUT / "data-audit.json") == protocol["data_audit_sha256"],
        "frozen_code_matches": digest(ROOT / "scripts/validate_external_desed.py") == protocol["code_sha256"],
        "frozen_model_unchanged": all(digest(ROOT / protocol["model_details"][f]) == protocol["model_details"][f + "_sha256"] for f in ("checkpoint", "onnx")),
        "old_release_evidence_unchanged": not changed,
        "tests_passed": tests["tests"] >= 24 and tests["failures"] == tests["errors"] == 0,
        "field_certification_not_claimed": report["physical_validation_performed"] is False and old_audit["field_release_complete"] is False,
    }
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
              "tests": tests, "old_release_evidence_files_checked": len(old_audit["evidence_sha256"]),
              "old_release_changed_paths": changed, "physical_validation_performed": False}
    write(OUT / "verification.json", result)
    if not all(checks.values()):
        raise ValueError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def archive(destination):
    verification = json.loads((OUT / "verification.json").read_text())
    if verification["status"] != "passed" or destination.exists():
        raise ValueError("Verification failed or archive already exists; no overwriting")
    destination.mkdir(parents=True)
    shutil.copytree(OUT, destination / "报告")
    shutil.copytree(DATA / "references", destination / "数据/references")
    for name in ("manifest.jsonl", "history-index.json"):
        shutil.copy2(DATA / name, destination / "数据" / name)
    for record in read_soundscape_manifest(DATA / "manifest.jsonl"):
        target = destination / "数据" / record.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DATA / record.path, target)
    sources = destination / "官方来源与授权"
    sources.mkdir()
    for p in [SOURCE / "desed-public-eval.official.tsv", SOURCE / "desed-public-eval.zenodo.json",
              SOURCE / "source-overlap-audit.json", DATA / "official/license_public_eval.tsv"]:
        shutil.copy2(p, sources / p.name)
    code = destination / "复现代码"
    code.mkdir()
    for name in ("audit_external_strong_sources.py", "validate_external_desed.py", "catalog_external_desed.py",
                 "summarize_external_desed.py", "finalize_external_desed.py"):
        shutil.copy2(ROOT / "scripts" / name, code / name)
    shutil.copy2(ROOT / "tests/test_external_desed_validation.py", code / "test_external_desed_validation.py")
    shutil.copy2(OUT / "结果解读.txt", destination / "先读这里.txt")
    files = sorted(p for p in destination.rglob("*") if p.is_file())
    inventory = {str(p.relative_to(destination)): digest(p) for p in files}
    write(destination / "SHA256SUMS.json", inventory)
    if not all(digest(destination / name) == sha for name, sha in inventory.items()):
        raise ValueError("Archive checksum verification failed")
    print(json.dumps({"archive": str(destination), "verified_files": len(files),
                      "bytes": sum(p.stat().st_size for p in files)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    archive(args.archive) if args.archive else verify()
