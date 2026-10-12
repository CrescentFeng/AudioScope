from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .audio import read_wav_mono
from .manifest import AudioRecord


def _torch_device(requested: str):
    import torch

    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _model_revision(model: Any) -> str:
    revision = getattr(model.config, "_commit_hash", None)
    return str(revision or "revision-unavailable")


def _embedding_signature(model_id: str, revision: str, family: str, records) -> str:
    digest = hashlib.sha256()
    digest.update(f"{family}:{model_id}:{revision}".encode("utf-8"))
    for record in records:
        digest.update(f"{record.asset_id}:{record.sha256}".encode("utf-8"))
    return digest.hexdigest()


def _load_complete_cache(cache_dir: Path, signature: str) -> Tuple[np.ndarray, dict] | None:
    metadata_path = cache_dir / "metadata.json"
    embeddings_path = cache_dir / "embeddings.npy"
    if not metadata_path.is_file() or not embeddings_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("signature") != signature or not metadata.get("complete"):
        return None
    embeddings = np.load(embeddings_path, allow_pickle=False)
    if embeddings.shape[0] != int(metadata["records"]):
        return None
    return embeddings.astype(np.float32, copy=False), metadata


def _save_embedding_cache(
    cache_dir: Path, embeddings: np.ndarray, metadata: Mapping[str, Any]
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "embeddings.npy", embeddings.astype(np.float32), allow_pickle=False)
    (cache_dir / "metadata.json").write_text(
        json.dumps(dict(metadata), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _batches(values: Sequence[Any], batch_size: int):
    for start in range(0, len(values), batch_size):
        yield start, values[start : start + batch_size]


def extract_ast_embeddings(
    records: List[AudioRecord],
    data_root: Path,
    settings: Mapping[str, Any],
    cache_dir: Path,
    seed: int,
) -> Tuple[np.ndarray, dict]:
    import torch
    from transformers import ASTModel, AutoFeatureExtractor

    _seed_everything(seed)
    torch.set_num_threads(int(settings.get("torch_threads", 8)))
    model_id = str(settings["model_id"])
    load_from = str(settings.get("model_path", model_id))
    device = _torch_device(str(settings.get("device", "auto")))
    local_only = bool(settings.get("local_files_only", False))
    processor = AutoFeatureExtractor.from_pretrained(load_from, local_files_only=local_only)
    model = ASTModel.from_pretrained(load_from, local_files_only=local_only).to(device).eval()
    revision = str(settings.get("pinned_revision") or _model_revision(model))
    signature = _embedding_signature(model_id, revision, "ast", records)
    cached = _load_complete_cache(cache_dir, signature)
    if cached is not None:
        return cached

    sampling_rate = int(processor.sampling_rate)
    batch_size = int(settings.get("batch_size", 4))
    rows = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start, batch in _batches(records, batch_size):
            waveforms = [read_wav_mono(data_root / item.path, sampling_rate)[0] for item in batch]
            inputs = processor(
                waveforms, sampling_rate=sampling_rate, return_tensors="pt", padding=True
            )
            outputs = model(**{key: value.to(device) for key, value in inputs.items()})
            pooled = outputs.pooler_output
            if pooled is None:
                pooled = outputs.last_hidden_state.mean(dim=1)
            rows.append(pooled.detach().cpu().numpy().astype(np.float32))
            if start == 0 or (start + len(batch)) % 100 == 0 or start + len(batch) == len(records):
                print(f"AST embeddings: {start + len(batch)}/{len(records)}", flush=True)
    embeddings = np.concatenate(rows, axis=0)
    metadata = {
        "complete": True,
        "family": "ast",
        "model_id": model_id,
        "model_revision": revision,
        "signature": signature,
        "records": len(records),
        "dimensions": int(embeddings.shape[1]),
        "sampling_rate": sampling_rate,
        "device": str(device),
        "batch_size": batch_size,
        "extraction_seconds": time.perf_counter() - started,
    }
    _save_embedding_cache(cache_dir, embeddings, metadata)
    return embeddings, metadata


def extract_clap_embeddings(
    records: List[AudioRecord],
    data_root: Path,
    settings: Mapping[str, Any],
    cache_dir: Path,
    seed: int,
) -> Tuple[np.ndarray, dict, Any, Any]:
    import torch
    from transformers import ClapModel, ClapProcessor

    _seed_everything(seed)
    torch.set_num_threads(int(settings.get("torch_threads", 8)))
    model_id = str(settings["model_id"])
    load_from = str(settings.get("model_path", model_id))
    device = _torch_device(str(settings.get("device", "auto")))
    local_only = bool(settings.get("local_files_only", False))
    processor = ClapProcessor.from_pretrained(load_from, local_files_only=local_only)
    model = ClapModel.from_pretrained(load_from, local_files_only=local_only).to(device).eval()
    revision = str(settings.get("pinned_revision") or _model_revision(model))
    signature = _embedding_signature(model_id, revision, "clap", records)
    cached = _load_complete_cache(cache_dir, signature)
    if cached is not None:
        return cached[0], cached[1], processor, model

    sampling_rate = int(processor.feature_extractor.sampling_rate)
    batch_size = int(settings.get("batch_size", 4))
    rows = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start, batch in _batches(records, batch_size):
            waveforms = [read_wav_mono(data_root / item.path, sampling_rate)[0] for item in batch]
            inputs = processor(
                audio=waveforms,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True,
            )
            audio_inputs = {
                key: value.to(device)
                for key, value in inputs.items()
                if key in {"input_features", "is_longer", "attention_mask"}
            }
            outputs = model.get_audio_features(**audio_inputs)
            rows.append(outputs.pooler_output.detach().cpu().numpy().astype(np.float32))
            if start == 0 or (start + len(batch)) % 100 == 0 or start + len(batch) == len(records):
                print(f"CLAP embeddings: {start + len(batch)}/{len(records)}", flush=True)
    embeddings = np.concatenate(rows, axis=0)
    metadata = {
        "complete": True,
        "family": "clap",
        "model_id": model_id,
        "model_revision": revision,
        "signature": signature,
        "records": len(records),
        "dimensions": int(embeddings.shape[1]),
        "sampling_rate": sampling_rate,
        "device": str(device),
        "batch_size": batch_size,
        "extraction_seconds": time.perf_counter() - started,
    }
    _save_embedding_cache(cache_dir, embeddings, metadata)
    return embeddings, metadata, processor, model


def select_few_shot_indices(
    records: Sequence[AudioRecord], train_indices: np.ndarray, count_per_class: int, seed: int
) -> np.ndarray:
    by_class: Dict[str, List[int]] = {}
    for index in train_indices.tolist():
        by_class.setdefault(records[index].label, []).append(index)
    selected = []
    for label, indices in sorted(by_class.items()):
        ranked = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"{seed}:{label}:{records[index].asset_id}".encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked) < count_per_class:
            raise ValueError(
                f"Class {label!r} has {len(ranked)} train records, fewer than {count_per_class}"
            )
        selected.extend(ranked[:count_per_class])
    return np.asarray(sorted(selected), dtype=np.int64)


def _new_probe(settings: Mapping[str, Any], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(settings.get("c", 1.0)),
                    max_iter=int(settings.get("max_iter", 2000)),
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def _metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
    }


def _write_predictions(
    report_dir: Path,
    records: Sequence[AudioRecord],
    class_names: Sequence[str],
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    metrics: Mapping[str, Any],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(
        json.dumps(dict(metrics), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    matrix = confusion_matrix(targets, predictions, labels=np.arange(len(class_names)))
    with (report_dir / "confusion_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *class_names])
        for label, row in zip(class_names, matrix):
            writer.writerow([label, *row.tolist()])
    with (report_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for record, target, prediction, scores in zip(
            records, targets, predictions, probabilities
        ):
            if target != prediction:
                handle.write(
                    json.dumps(
                        {
                            "asset_id": record.asset_id,
                            "path": record.path,
                            "actual": class_names[int(target)],
                            "predicted": class_names[int(prediction)],
                            "confidence": float(scores[int(prediction)]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def evaluate_linear_probe(
    records: List[AudioRecord],
    embeddings: np.ndarray,
    folds: Iterable[int],
    settings: Mapping[str, Any],
    seed: int,
    artifact_root: Path,
    report_root: Path,
    sample_counts: Iterable[int] = (),
) -> dict:
    folds = [int(fold) for fold in folds]
    class_names = sorted({record.label for record in records})
    class_to_index = {label: index for index, label in enumerate(class_names)}
    labels = np.asarray([class_to_index[record.label] for record in records], dtype=np.int64)
    fold_ids = np.asarray([record.fold for record in records], dtype=np.int64)
    fold_results = []
    for fold in folds:
        train_indices = np.flatnonzero(fold_ids != fold)
        test_indices = np.flatnonzero(fold_ids == fold)
        probe = _new_probe(settings, seed + fold)
        started = time.perf_counter()
        probe.fit(embeddings[train_indices], labels[train_indices])
        training_seconds = time.perf_counter() - started
        prediction_started = time.perf_counter()
        predictions = probe.predict(embeddings[test_indices])
        probabilities = probe.predict_proba(embeddings[test_indices])
        inference_seconds = time.perf_counter() - prediction_started
        result = {
            "fold": fold,
            "train_records": len(train_indices),
            "test_records": len(test_indices),
            "training_seconds": training_seconds,
            "probe_inference_seconds": inference_seconds,
            **_metrics(labels[test_indices], predictions),
        }
        fold_results.append(result)
        artifact_dir = artifact_root / f"fold-{fold}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "probe": probe,
                "class_names": class_names,
                "embedding_dimensions": embeddings.shape[1],
            },
            artifact_dir / "linear_probe.joblib",
        )
        _write_predictions(
            report_root / f"fold-{fold}",
            [records[index] for index in test_indices],
            class_names,
            labels[test_indices],
            predictions,
            probabilities,
            result,
        )

    aggregate = {}
    for metric in ("accuracy", "macro_f1"):
        values = [row[metric] for row in fold_results]
        aggregate[metric] = {"mean": mean(values), "std": pstdev(values), "values": values}
    aggregate["training_seconds_total"] = sum(row["training_seconds"] for row in fold_results)

    sample_efficiency = []
    for count in [int(value) for value in sample_counts]:
        values = []
        training_records = []
        elapsed = 0.0
        for fold in folds:
            train_indices = select_few_shot_indices(
                records, np.flatnonzero(fold_ids != fold), count, seed + fold
            )
            test_indices = np.flatnonzero(fold_ids == fold)
            probe = _new_probe(settings, seed + 1000 + fold + count)
            started = time.perf_counter()
            probe.fit(embeddings[train_indices], labels[train_indices])
            predictions = probe.predict(embeddings[test_indices])
            elapsed += time.perf_counter() - started
            values.append(_metrics(labels[test_indices], predictions))
            training_records.append(len(train_indices))
        sample_efficiency.append(
            {
                "samples_per_class": count,
                "training_records_per_fold": training_records,
                "accuracy_mean": mean(row["accuracy"] for row in values),
                "accuracy_std": pstdev(row["accuracy"] for row in values),
                "macro_f1_mean": mean(row["macro_f1"] for row in values),
                "macro_f1_std": pstdev(row["macro_f1"] for row in values),
                "training_seconds_total": elapsed,
                "fold_metrics": values,
            }
        )
    return {
        "protocol": "frozen-embedding-linear-probe-official-five-fold",
        "folds": fold_results,
        "aggregate": aggregate,
        "sample_efficiency": sample_efficiency,
    }


def humanize_label(label: str) -> str:
    replacements = {
        "can_opening": "opening a beverage can",
        "clock_alarm": "an alarm clock ringing",
        "clock_tick": "a clock ticking",
        "door_wood_creaks": "a wooden door creaking",
        "door_wood_knock": "knocking on a wooden door",
        "hand_saw": "a hand saw cutting wood",
        "mouse_click": "a computer mouse clicking",
        "pouring_water": "water being poured",
        "washing_machine": "a washing machine running",
    }
    return replacements.get(label, label.replace("_", " "))


def build_clap_text_embeddings(
    class_names: Sequence[str],
    templates: Sequence[str],
    processor: Any,
    model: Any,
    device: Any,
) -> Tuple[np.ndarray, List[List[str]]]:
    import torch

    prompt_groups = [
        [template.format(label=humanize_label(label)) for template in templates]
        for label in class_names
    ]
    prompts = [prompt for group in prompt_groups for prompt in group]
    inputs = processor(text=prompts, padding=True, return_tensors="pt")
    text_inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if key in {"input_ids", "attention_mask", "position_ids"}
    }
    with torch.inference_mode():
        outputs = model.get_text_features(**text_inputs).pooler_output
        rows = outputs.reshape(len(class_names), len(templates), -1).mean(dim=1)
        rows = torch.nn.functional.normalize(rows, dim=-1)
    return rows.detach().cpu().numpy().astype(np.float32), prompt_groups


def evaluate_zero_shot(
    records: List[AudioRecord],
    audio_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    prompt_groups: List[List[str]],
    folds: Iterable[int],
    report_root: Path,
) -> dict:
    class_names = sorted({record.label for record in records})
    class_to_index = {label: index for index, label in enumerate(class_names)}
    labels = np.asarray([class_to_index[record.label] for record in records], dtype=np.int64)
    normalized_audio = audio_embeddings / np.maximum(
        np.linalg.norm(audio_embeddings, axis=1, keepdims=True), 1e-12
    )
    similarities = normalized_audio @ text_embeddings.T
    exp_scores = np.exp(similarities - similarities.max(axis=1, keepdims=True))
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    predictions = similarities.argmax(axis=1)
    fold_results = []
    fold_ids = np.asarray([record.fold for record in records], dtype=np.int64)
    for fold in [int(value) for value in folds]:
        indices = np.flatnonzero(fold_ids == fold)
        result = {
            "fold": fold,
            "test_records": len(indices),
            **_metrics(labels[indices], predictions[indices]),
        }
        fold_results.append(result)
        _write_predictions(
            report_root / f"fold-{fold}",
            [records[index] for index in indices],
            class_names,
            labels[indices],
            predictions[indices],
            probabilities[indices],
            result,
        )
    aggregate = {}
    for metric in ("accuracy", "macro_f1"):
        values = [row[metric] for row in fold_results]
        aggregate[metric] = {"mean": mean(values), "std": pstdev(values), "values": values}
    result = {
        "protocol": "clap-zero-shot-fixed-prompt-ensemble",
        "prompt_groups": dict(zip(class_names, prompt_groups)),
        "folds": fold_results,
        "aggregate": aggregate,
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "zero_shot.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def run_ast_transfer(
    records: List[AudioRecord],
    data_root: Path,
    folds: Iterable[int],
    transfer_settings: Mapping[str, Any],
    probe_settings: Mapping[str, Any],
    seed: int,
    cache_dir: Path,
    artifact_root: Path,
    report_root: Path,
) -> dict:
    embeddings, metadata = extract_ast_embeddings(
        records, data_root, transfer_settings, cache_dir, seed
    )
    probe = evaluate_linear_probe(
        records,
        embeddings,
        folds,
        probe_settings,
        seed,
        artifact_root,
        report_root / "linear-probe",
        probe_settings.get("sample_counts_per_class", ()),
    )
    result = {"encoder": metadata, "linear_probe": probe}
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "transfer_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def run_clap_transfer(
    records: List[AudioRecord],
    data_root: Path,
    folds: Iterable[int],
    transfer_settings: Mapping[str, Any],
    probe_settings: Mapping[str, Any],
    zero_shot_settings: Mapping[str, Any],
    seed: int,
    cache_dir: Path,
    artifact_root: Path,
    report_root: Path,
) -> dict:
    embeddings, metadata, processor, model = extract_clap_embeddings(
        records, data_root, transfer_settings, cache_dir, seed
    )
    device = _torch_device(str(transfer_settings.get("device", "auto")))
    class_names = sorted({record.label for record in records})
    text_embeddings, prompt_groups = build_clap_text_embeddings(
        class_names,
        list(zero_shot_settings["prompt_templates"]),
        processor,
        model,
        device,
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    np.save(artifact_root / "text_embeddings.npy", text_embeddings, allow_pickle=False)
    (artifact_root / "prompts.json").write_text(
        json.dumps(dict(zip(class_names, prompt_groups)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    zero_shot = evaluate_zero_shot(
        records,
        embeddings,
        text_embeddings,
        prompt_groups,
        folds,
        report_root / "zero-shot",
    )
    linear_probe = evaluate_linear_probe(
        records,
        embeddings,
        folds,
        probe_settings,
        seed,
        artifact_root / "linear-probe",
        report_root / "linear-probe",
    )
    result = {"encoder": metadata, "zero_shot": zero_shot, "linear_probe": linear_probe}
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "transfer_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result
