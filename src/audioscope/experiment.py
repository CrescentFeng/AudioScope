from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .audio import add_white_noise, read_wav_mono
from .features import extract_mfcc_summary
from .manifest import AudioRecord


def _matrix(
    records: Iterable[AudioRecord],
    data_root: Path,
    sample_rate: int,
    feature_settings: Mapping[str, int],
    snr_db: Optional[float] = None,
    seed: int = 0,
    cache_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, List[AudioRecord]]:
    rows, labels, ordered = [], [], []
    for index, record in enumerate(records):
        cache_path = None
        if cache_dir is not None and snr_db is None:
            feature_signature = json.dumps(
                {"sample_rate": sample_rate, **dict(feature_settings)}, sort_keys=True
            )
            cache_key = hashlib.sha256(
                f"{record.sha256}:{feature_signature}".encode("utf-8")
            ).hexdigest()
            cache_path = cache_dir / f"{cache_key}.npy"
        if cache_path is not None and cache_path.is_file():
            features = np.load(cache_path, allow_pickle=False)
        else:
            samples, _ = read_wav_mono(data_root / record.path, sample_rate)
            if snr_db is not None:
                samples = add_white_noise(samples, snr_db, seed + index)
            features = extract_mfcc_summary(samples, sample_rate, feature_settings)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, features, allow_pickle=False)
        rows.append(features)
        labels.append(record.label)
        ordered.append(record)
    return np.stack(rows), np.asarray(labels), ordered


def train(
    records: List[AudioRecord],
    data_root: Path,
    sample_rate: int,
    feature_settings: Mapping[str, int],
    model_settings: Mapping[str, object],
    test_fold: int,
    seed: int,
    output_path: Path,
    cache_dir: Optional[Path] = None,
) -> dict:
    train_records = [record for record in records if record.fold != test_fold]
    if not train_records:
        raise ValueError("Training split is empty")
    x_train, y_train, _ = _matrix(
        train_records,
        data_root,
        sample_rate,
        feature_settings,
        cache_dir=cache_dir,
    )
    base_estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                SVC(
                    C=float(model_settings.get("c", 10.0)),
                    kernel=str(model_settings.get("kernel", "rbf")),
                    random_state=seed,
                ),
            ),
        ]
    )
    estimator = CalibratedClassifierCV(
        base_estimator,
        method="sigmoid",
        cv=3,
        ensemble=False,
    )
    started = time.perf_counter()
    estimator.fit(x_train, y_train)
    training_seconds = time.perf_counter() - started
    artifact = {
        "estimator": estimator,
        "sample_rate": sample_rate,
        "feature_settings": dict(feature_settings),
        "classes": estimator.classes_.tolist(),
        "test_fold": test_fold,
        "seed": seed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return {
        "training_records": len(train_records),
        "classes": artifact["classes"],
        "training_seconds": training_seconds,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "scikit_learn": sklearn.__version__,
    }


def load_artifact(path: Path) -> dict:
    artifact = joblib.load(path)
    required = {"estimator", "sample_rate", "feature_settings", "classes", "test_fold"}
    if not required.issubset(artifact):
        raise ValueError(f"Invalid model artifact, missing: {sorted(required - set(artifact))}")
    return artifact


def predict_file(path: Path, artifact: dict) -> List[dict]:
    samples, _ = read_wav_mono(path, int(artifact["sample_rate"]))
    features = extract_mfcc_summary(
        samples, int(artifact["sample_rate"]), artifact["feature_settings"]
    )
    probabilities = artifact["estimator"].predict_proba(features.reshape(1, -1))[0]
    ranked = np.argsort(probabilities)[::-1]
    return [
        {
            "label": str(artifact["estimator"].classes_[index]),
            "probability": float(probabilities[index]),
        }
        for index in ranked
    ]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate(
    records: List[AudioRecord],
    data_root: Path,
    artifact: dict,
    report_dir: Path,
    snr_values: Iterable[float],
    seed: int,
    cache_dir: Optional[Path] = None,
) -> dict:
    test_records = [record for record in records if record.fold == int(artifact["test_fold"])]
    if not test_records:
        raise ValueError("Test split is empty")
    x_test, y_true, ordered = _matrix(
        test_records,
        data_root,
        int(artifact["sample_rate"]),
        artifact["feature_settings"],
        cache_dir=cache_dir,
    )
    started = time.perf_counter()
    y_pred = artifact["estimator"].predict(x_test)
    inference_seconds = time.perf_counter() - started
    labels = list(artifact["classes"])
    result = _metrics(y_true, y_pred)
    result.update(
        {
            "test_fold": int(artifact["test_fold"]),
            "test_records": len(test_records),
            "inference_seconds": inference_seconds,
            "mean_seconds_per_clip": inference_seconds / len(test_records),
            "robustness": [],
            "scope": "synthetic-smoke"
            if all(r.dataset == "audioscope-smoke" for r in test_records)
            else "dataset-evaluation",
        }
    )
    for snr_db in snr_values:
        noisy_x, noisy_y, _ = _matrix(
            test_records,
            data_root,
            int(artifact["sample_rate"]),
            artifact["feature_settings"],
            float(snr_db),
            seed,
        )
        noisy_pred = artifact["estimator"].predict(noisy_x)
        result["robustness"].append({"snr_db": float(snr_db), **_metrics(noisy_y, noisy_pred)})
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    with (report_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])
    with (report_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for record, actual, predicted in zip(ordered, y_true, y_pred):
            if actual != predicted:
                handle.write(
                    json.dumps(
                        {
                            "asset_id": record.asset_id,
                            "path": record.path,
                            "actual": str(actual),
                            "predicted": str(predicted),
                        }
                    )
                    + "\n"
                )
    return result
