from __future__ import annotations

from pathlib import Path

import numpy as np

from audioscope.manifest import AudioRecord
from audioscope.transfer import evaluate_linear_probe, humanize_label, select_few_shot_indices


def _records() -> list[AudioRecord]:
    records = []
    for fold in (1, 2):
        for label_index, label in enumerate(("cat", "dog")):
            for example in range(3):
                asset = f"{fold}-{label}-{example}"
                records.append(
                    AudioRecord(
                        asset_id=asset,
                        path=f"{asset}.wav",
                        sha256=asset,
                        dataset="synthetic",
                        dataset_version="1",
                        source_url="local",
                        license="CC0-1.0",
                        sample_rate=16000,
                        channels=1,
                        duration_s=1.0,
                        label=label,
                        fold=fold,
                        source_recording_id=asset,
                    )
                )
    return records


def test_few_shot_selection_is_balanced_and_deterministic():
    records = _records()
    train = np.flatnonzero(np.asarray([record.fold for record in records]) == 1)
    first = select_few_shot_indices(records, train, 2, 2027)
    second = select_few_shot_indices(records, train, 2, 2027)
    assert first.tolist() == second.tolist()
    assert len(first) == 4
    assert sorted(records[index].label for index in first) == ["cat", "cat", "dog", "dog"]


def test_linear_probe_smoke(tmp_path: Path):
    records = _records()
    embeddings = np.asarray(
        [
            [-2.0 + 0.01 * index, 1.0] if record.label == "cat" else [2.0 + 0.01 * index, -1.0]
            for index, record in enumerate(records)
        ],
        dtype=np.float32,
    )
    result = evaluate_linear_probe(
        records,
        embeddings,
        folds=[1, 2],
        settings={"c": 1.0, "max_iter": 200},
        seed=2027,
        artifact_root=tmp_path / "artifacts",
        report_root=tmp_path / "reports",
        sample_counts=[1, 2],
    )
    assert result["aggregate"]["macro_f1"]["mean"] == 1.0
    assert len(result["sample_efficiency"]) == 2
    assert (tmp_path / "artifacts" / "fold-1" / "linear_probe.joblib").is_file()


def test_humanize_label_handles_special_cases():
    assert humanize_label("door_wood_creaks") == "a wooden door creaking"
    assert humanize_label("dog") == "dog"
