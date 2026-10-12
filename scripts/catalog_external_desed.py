"""Add external test metadata/evidence to PostgreSQL; never change release gates."""
import json
from datetime import datetime, timezone

from sqlalchemy import func, select
from validate_external_desed import DATA, OUT, ROOT, digest, write

from audioscope.catalog_import import _stable_id
from audioscope.db import (
    Annotation,
    AudioFile,
    BenchmarkResult,
    Dataset,
    DatasetSplit,
    DatasetVersion,
    License,
    ProcessingRun,
    Source,
)
from audioscope.release_catalog import sessions


def main():
    report_path = OUT / "evaluation.json"
    report = json.loads(report_path.read_text())
    audit = json.loads((OUT / "data-audit.json").read_text())
    if report["status"] != "completed" or report["physical_validation_performed"] is not False:
        raise ValueError("Not an eligible external-test evidence report")
    manifest = DATA / "manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    integrity = {r["filename"]: r for r in audit["audio_integrity"]}
    manifest_hash = digest(manifest)
    slug = "desed-public-external-test"
    version_id = _stable_id(slug, manifest_hash)
    run_id = _stable_id("external-strong-evaluation", digest(report_path))
    metric_count = 0
    with sessions(ROOT)() as session:
        source = session.scalar(select(Source).where(Source.name == "DESED public evaluation (Zenodo 4560759)",
                                                     Source.source_url == "https://zenodo.org/records/4560759"))
        if source is None:
            source = Source(name="DESED public evaluation (Zenodo 4560759)",
                            source_url="https://zenodo.org/records/4560759",
                            citation="Turpault, Serizel, Shah and Salamon. DESED_public_eval. doi:10.5281/zenodo.4560759")
            session.add(source)
            session.flush()
        license_row = session.scalar(select(License).where(License.spdx_or_name == "CC-BY-4.0"))
        if license_row is None:
            license_row = License(spdx_or_name="CC-BY-4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
                                  notes="License declared by upstream; preserve upstream attribution.")
            session.add(license_row)
            session.flush()
        dataset = session.scalar(select(Dataset).where(Dataset.slug == slug))
        if dataset is None:
            dataset = Dataset(slug=slug, display_name="DESED public — external real-recording test",
                              task_type="sound_event_detection", source_id=source.id, license_id=license_row.id,
                              description="Official strong human annotations. External test only; no self-owned field certification.")
            session.add(dataset)
            session.flush()
        session.merge(DatasetVersion(id=version_id, dataset_id=dataset.id,
                      version_key=f"zenodo4560759-{manifest_hash[:16]}",
                      manifest_uri="audioscope-data://external-desed-public/manifest.jsonl",
                      manifest_sha256=manifest_hash, record_count=len(records),
                      metadata_json={"test_only": True, "training_eligible": False,
                                     "annotation_sha256": audit["annotation_sha256"],
                                     "data_audit_uri": "audioscope://reports/generated/external-desed-public/data-audit.json",
                                     "physical_validation_performed": False}))
        session.flush()
        audio_ids = []
        for r in records:
            path = DATA / r["path"]
            info = integrity[path.name]
            reference = json.loads((DATA / "references" / f"{r['scene_id']}.reference.json").read_text())
            if digest(path) != info["sha256"]:
                raise ValueError(f"Audio changed: {path.name}")
            audio_id = _stable_id(version_id, r["scene_id"])
            audio_ids.append(audio_id)
            session.merge(AudioFile(id=audio_id, dataset_version_id=version_id,
                          storage_uri=f"audioscope-data://external-desed-public/{r['path']}",
                          sha256=info["sha256"], byte_size=path.stat().st_size, duration_s=r["duration_s"],
                          sample_rate=info["sample_rate"], channels=info["channels"], codec="wav",
                          source_recording_id=r["events"][0]["source_recording_id"],
                          metadata_json={"test_only": True, "training_eligible": False,
                                         "reference_source": "official_human_strong_annotations",
                                         "upstream_attribution": reference["upstream_attribution"],
                                         "reference_uri": f"audioscope-data://external-desed-public/references/{r['scene_id']}.reference.json"}))
            session.flush()
            session.merge(DatasetSplit(id=_stable_id(audio_id, "external_test"), dataset_version_id=version_id,
                          audio_file_id=audio_id, split_name="external_test",
                          protocol="Frozen current default model; no training or threshold tuning"))
            for index, e in enumerate(r["events"]):
                session.merge(Annotation(id=_stable_id(audio_id, "official", index), audio_file_id=audio_id,
                              label=e["label"], onset_s=e["onset_s"], offset_s=e["offset_s"],
                              annotation_type="strong", annotator="DESED official human annotation",
                              metadata_json={"source": "dataset_reference", "annotation_sha256": audit["annotation_sha256"],
                                             "assistant_generated": False}))
        session.merge(ProcessingRun(id=run_id, run_kind="external_human_strong_evaluation", status="completed",
                      finished_at=datetime.now(timezone.utc),
                      config_uri="audioscope://reports/generated/external-desed-public/frozen-protocol.json",
                      environment_json={"physical_validation_performed": False},
                      summary_json={"report_uri": "audioscope://reports/generated/external-desed-public/evaluation.json",
                                    "report_sha256": digest(report_path), "dataset_version_id": version_id,
                                    "model": report["model"], "test_only": True}))
        session.flush()
        for kind in ("event_based", "segment_based"):
            group = report["metrics"][kind]
            for slice_name, scores in {"micro": group["micro"], **group["per_class"]}.items():
                for metric in ("precision", "recall", "f1", "tp", "fp", "fn"):
                    name = f"{kind}.{metric}"
                    session.merge(BenchmarkResult(id=_stable_id(run_id, name, slice_name), run_id=run_id,
                                  metric_name=name, metric_value=float(scores[metric]), slice_name=slice_name,
                                  metadata_json={"dataset_version_id": version_id, "external_test": True}))
                    metric_count += 1
            session.merge(BenchmarkResult(id=_stable_id(run_id, kind, "macro_f1"), run_id=run_id,
                          metric_name=f"{kind}.macro_f1", metric_value=float(group["macro_f1"]), slice_name="macro",
                          metadata_json={"dataset_version_id": version_id, "external_test": True}))
            metric_count += 1
        session.flush()
        counts = {
            "audio_files": session.scalar(select(func.count()).select_from(AudioFile).where(AudioFile.dataset_version_id == version_id)),
            "annotations": session.scalar(select(func.count()).select_from(Annotation).where(Annotation.audio_file_id.in_(audio_ids))),
            "benchmark_results": session.scalar(select(func.count()).select_from(BenchmarkResult).where(BenchmarkResult.run_id == run_id))}
        if counts != {"audio_files": len(records), "annotations": report["reference_events"], "benchmark_results": metric_count}:
            raise ValueError(f"Catalog count mismatch: {counts}")
        session.commit()
    write(OUT / "catalog-import.json", {"status": "passed", "dataset_version_id": version_id,
          "processing_run_id": run_id, "counts": counts, "report_sha256": digest(report_path),
          "existing_release_audits_modified": False, "physical_validation_performed": False})
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
