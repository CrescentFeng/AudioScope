"""Audit official external annotations only; does not run inference or training."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/external-strong-source-audit"
COMMIT = "0c57c266c19079358d2a18b524f1f74f878fc6fc"


def main():
    path = OUT / "desed-public-eval.official.tsv"
    with path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    known = defaultdict(list)
    manifest_paths = ["data/desed-real-strong/manifest.jsonl",
                      "data/desed-real-strong-only/manifest.jsonl"]
    for name in manifest_paths:
        for r in map(json.loads, (ROOT / name).read_text().splitlines()):
            # Project IDs have an added Y prefix; the next 11 chars are YouTube ID.
            assert r["scene_id"].startswith("Y")
            source_id = r["scene_id"][1:12]
            known[source_id].append({"manifest": name, "scene_id": r["scene_id"], "split": r["split"]})
    groups = defaultdict(list)
    for row in rows:
        start, end = float(row["onset"]), float(row["offset"])
        assert math.isfinite(start) and math.isfinite(end) and 0 <= start < end
        groups[row["filename"]].append({"label": row["event_label"].lower(),
                                        "onset_s": start, "offset_s": end})
    retained, excluded, duration_checks = [], [], []
    for name, events in sorted(groups.items()):
        video_id, start, end = Path(name).stem.rsplit("_", 2)
        assert len(video_id) == 11
        nominal_duration = float(end) - float(start)
        if nominal_duration != 10 or max(e["offset_s"] for e in events) > nominal_duration:
            duration_checks.append({"filename": name, "filename_duration_s": nominal_duration,
                                    "maximum_annotation_offset_s": max(e["offset_s"] for e in events),
                                    "action": "核对实际音频时长；不得一律裁成10秒或擅自截断原标注。"})
        if video_id in known:
            excluded.append({"filename": name, "source_recording_id": video_id,
                             "existing_uses": known[video_id], "reason": "same_source_video_even_if_different_excerpt"})
        else:
            retained.append({"filename": name, "source_recording_id": video_id,
                             "events": events, "annotation_source": "official_human_strong_annotations",
                             "annotation_revision": COMMIT, "source": "dataset_reference",
                             "complete": False, "audio_downloaded": False,
                             "formal_evaluation_eligible": False, "training_eligible": False,
                             "status": "candidate_pending_audio_integrity_license_and_alignment_checks"})
    report = {"official_annotation_url": f"https://github.com/turpaultn/DESED/blob/{COMMIT}/data/real/metadata/eval/public.tsv",
              "official_audio_record": "https://zenodo.org/records/4560759",
              "annotation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "official_recordings": len(groups), "official_events": len(rows),
              "project_manifests_checked": manifest_paths,
              "project_unique_video_ids_checked": len(known),
              "excluded": excluded, "retained_recordings": len(retained),
              "retained_events": sum(len(r["events"]) for r in retained),
              "retained_class_event_counts": dict(sorted(Counter(e["label"] for r in retained for e in r["events"]).items())),
              "duration_checks_pending": duration_checks,
              "audio_downloaded": False, "waveform_duplicates_checked": False,
              "pretraining_overlap": "unknown", "model_inference_run": False,
              "physical_validation_performed": False,
              "interpretation": "找到官方人工强标注及可下载音频；候选集与项目登记的视频ID去重，不等于完成波形去重或完成准确率验证。"}
    (OUT / "source-overlap-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (OUT / "candidate-manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in retained))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
