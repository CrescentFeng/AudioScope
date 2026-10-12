from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .audio import add_white_noise, read_wav_mono
from .features import extract_logmel
from .manifest import AudioRecord


class LogMelCNN(nn.Module):
    def __init__(self, num_classes: int, channels: Sequence[int], dropout: float) -> None:
        super().__init__()
        blocks = []
        input_channels = 1
        for output_channels in channels:
            blocks.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            input_channels = output_channels
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((4, 1))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(input_channels * 4, num_classes)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.encoder(inputs)))


class SpectrogramDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        normalization: Tuple[float, float],
        augmentation: Optional[Mapping[str, int]] = None,
        feature_variants: Optional[Sequence[np.ndarray]] = None,
        variant_probability: float = 0.0,
    ) -> None:
        self.features = features
        self.labels = labels
        self.feature_mean, self.feature_std = normalization
        self.augmentation = augmentation
        self.feature_variants = feature_variants or []
        self.variant_probability = variant_probability

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        source = self.features
        if self.feature_variants and torch.rand(1).item() < self.variant_probability:
            variant_index = int(torch.randint(0, len(self.feature_variants), (1,)).item())
            source = self.feature_variants[variant_index]
        feature = torch.from_numpy(source[index].copy()).unsqueeze(0)
        feature = (feature - self.feature_mean) / self.feature_std
        if self.augmentation:
            max_shift = int(self.augmentation.get("max_time_shift_frames", 0))
            if max_shift > 0:
                shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
                feature = torch.roll(feature, shifts=shift, dims=2)
            frequency_mask = int(self.augmentation.get("frequency_mask", 0))
            if frequency_mask > 0:
                width = int(torch.randint(0, frequency_mask + 1, (1,)).item())
                if width:
                    start = int(torch.randint(0, feature.shape[1] - width + 1, (1,)).item())
                    feature[:, start : start + width, :] = 0.0
            time_mask = int(self.augmentation.get("time_mask", 0))
            if time_mask > 0:
                width = int(torch.randint(0, time_mask + 1, (1,)).item())
                if width:
                    start = int(torch.randint(0, feature.shape[2] - width + 1, (1,)).item())
                    feature[:, :, start : start + width] = 0.0
        return feature, torch.tensor(self.labels[index], dtype=torch.long)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _device(requested: str) -> torch.device:
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


def _cache_path(
    cache_dir: Path,
    record: AudioRecord,
    sample_rate: int,
    settings: Mapping,
    variant: str = "clean",
) -> Path:
    signature = json.dumps(
        {"sample_rate": sample_rate, "variant": variant, **dict(settings)}, sort_keys=True
    )
    key = hashlib.sha256(f"{record.sha256}:{signature}".encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.npy"


def build_logmel_matrix(
    records: Iterable[AudioRecord],
    data_root: Path,
    sample_rate: int,
    settings: Mapping[str, int],
    cache_dir: Optional[Path] = None,
    snr_db: Optional[float] = None,
    seed: int = 0,
    cache_noisy: bool = False,
) -> np.ndarray:
    rows = []
    for index, record in enumerate(records):
        cached = None
        if cache_dir is not None and (snr_db is None or cache_noisy):
            variant = "clean" if snr_db is None else f"snr-{snr_db}-seed-{seed + index}"
            cached = _cache_path(cache_dir, record, sample_rate, settings, variant)
        if cached is not None and cached.is_file():
            feature = np.load(cached, allow_pickle=False)
        else:
            samples, _ = read_wav_mono(data_root / record.path, sample_rate)
            if snr_db is not None:
                samples = add_white_noise(samples, snr_db, seed + index)
            feature = extract_logmel(samples, sample_rate, settings)
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                np.save(cached, feature, allow_pickle=False)
        rows.append(feature)
    return np.stack(rows).astype(np.float32)


def _predict(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    normalization: Tuple[float, float],
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    dataset = SpectrogramDataset(features, labels, normalization)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions = []
    probabilities = []
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for batch, _ in loader:
            logits = model(batch.to(device))
            batch_probabilities = torch.softmax(logits, dim=1)
            probabilities.append(batch_probabilities.cpu().numpy())
            predictions.append(batch_probabilities.argmax(dim=1).cpu().numpy())
    elapsed = time.perf_counter() - started
    return np.concatenate(predictions), np.concatenate(probabilities), elapsed


def _metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
    }


def _write_fold_report(
    report_dir: Path,
    records: List[AudioRecord],
    class_names: List[str],
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    result: dict,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    matrix = confusion_matrix(targets, predictions, labels=np.arange(len(class_names)))
    with (report_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *class_names])
        for label, row in zip(class_names, matrix):
            writer.writerow([label, *row.tolist()])
    with (report_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for record, target, prediction, scores in zip(records, targets, predictions, probabilities):
            if target != prediction:
                handle.write(
                    json.dumps(
                        {
                            "asset_id": record.asset_id,
                            "path": record.path,
                            "actual": class_names[int(target)],
                            "predicted": class_names[int(prediction)],
                            "confidence": float(scores[int(prediction)]),
                        }
                    )
                    + "\n"
                )


def _train_fold(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    class_names: List[str],
    settings: Mapping,
    seed: int,
    device: torch.device,
    feature_variants: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[LogMelCNN, Tuple[float, float], dict]:
    _set_seed(seed)
    train_features = features[train_indices]
    normalization = (float(train_features.mean()), float(train_features.std() + 1e-6))
    dataset = SpectrogramDataset(
        train_features,
        labels[train_indices],
        normalization,
        settings.get("augmentation"),
        [variant[train_indices] for variant in feature_variants or []],
        float(settings.get("waveform_noise_probability", 0.0)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    model = LogMelCNN(
        len(class_names),
        [int(value) for value in settings["channels"]],
        float(settings["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(settings["label_smoothing"]))
    history = []
    started = time.perf_counter()
    for epoch in range(int(settings["epochs"])):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for batch, targets in loader:
            batch = batch.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(targets)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            seen += len(targets)
        history.append(
            {"epoch": epoch + 1, "loss": total_loss / seen, "training_accuracy": correct / seen}
        )
    training_seconds = time.perf_counter() - started
    details = {
        "epochs": int(settings["epochs"]),
        "training_records": len(train_indices),
        "training_seconds": training_seconds,
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
    }
    return model, normalization, details


def run_cnn_cross_validation(
    records: List[AudioRecord],
    data_root: Path,
    sample_rate: int,
    feature_settings: Mapping[str, int],
    model_settings: Mapping,
    folds: Iterable[int],
    snr_values: Iterable[float],
    seed: int,
    artifact_root: Path,
    report_root: Path,
    cache_dir: Optional[Path] = None,
) -> dict:
    folds = [int(fold) for fold in folds]
    class_names = sorted({record.label for record in records})
    class_to_index = {label: index for index, label in enumerate(class_names)}
    labels = np.asarray([class_to_index[record.label] for record in records], dtype=np.int64)
    fold_ids = np.asarray([record.fold for record in records], dtype=np.int64)
    features = build_logmel_matrix(
        records, data_root, sample_rate, feature_settings, cache_dir=cache_dir
    )
    noise_variants = []
    for variant_index, snr_db in enumerate(model_settings.get("waveform_noise_snr_db", [])):
        noise_variants.append(
            build_logmel_matrix(
                records,
                data_root,
                sample_rate,
                feature_settings,
                cache_dir=cache_dir,
                snr_db=float(snr_db),
                seed=seed + (variant_index + 1) * 100_000,
                cache_noisy=True,
            )
        )
    device = _device(str(model_settings.get("device", "auto")))
    fold_results = []
    for fold in folds:
        train_indices = np.flatnonzero(fold_ids != fold)
        test_indices = np.flatnonzero(fold_ids == fold)
        test_records = [records[index] for index in test_indices]
        model, normalization, training = _train_fold(
            features,
            labels,
            train_indices,
            class_names,
            model_settings,
            seed + fold,
            device,
            noise_variants,
        )
        predictions, probabilities, inference_seconds = _predict(
            model,
            features[test_indices],
            labels[test_indices],
            normalization,
            int(model_settings["batch_size"]),
            device,
        )
        result = {
            "test_fold": fold,
            "test_records": len(test_indices),
            **_metrics(labels[test_indices], predictions),
            "inference_seconds_logmel_only": inference_seconds,
            "robustness": [],
        }
        for snr_db in snr_values:
            noisy_features = build_logmel_matrix(
                test_records,
                data_root,
                sample_rate,
                feature_settings,
                snr_db=float(snr_db),
                seed=seed + fold * 1000,
            )
            noisy_predictions, _, _ = _predict(
                model,
                noisy_features,
                labels[test_indices],
                normalization,
                int(model_settings["batch_size"]),
                device,
            )
            result["robustness"].append(
                {"snr_db": float(snr_db), **_metrics(labels[test_indices], noisy_predictions)}
            )
        artifact_dir = artifact_root / f"fold-{fold}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.cpu().state_dict(),
                "class_names": class_names,
                "channels": [int(value) for value in model_settings["channels"]],
                "dropout": float(model_settings["dropout"]),
                "normalization": normalization,
                "sample_rate": sample_rate,
                "feature_settings": dict(feature_settings),
                "test_fold": fold,
                "seed": seed + fold,
            },
            artifact_dir / "model.pt",
        )
        (artifact_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "class_names": class_names,
                    "normalization": normalization,
                    "sample_rate": sample_rate,
                    "feature_settings": dict(feature_settings),
                    "test_fold": fold,
                    "training": training,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        _write_fold_report(
            report_root / f"fold-{fold}",
            test_records,
            class_names,
            labels[test_indices],
            predictions,
            probabilities,
            {"training": training, **result},
        )
        fold_results.append({"fold": fold, "training": training, "metrics": result})
    aggregate = {}
    for metric in ("accuracy", "macro_f1"):
        values = [item["metrics"][metric] for item in fold_results]
        aggregate[metric] = {"mean": mean(values), "std": pstdev(values), "values": values}
    aggregate["robustness"] = []
    for snr_db in snr_values:
        rows = [
            next(row for row in item["metrics"]["robustness"] if row["snr_db"] == float(snr_db))
            for item in fold_results
        ]
        aggregate["robustness"].append(
            {
                "snr_db": float(snr_db),
                "accuracy_mean": mean(row["accuracy"] for row in rows),
                "macro_f1_mean": mean(row["macro_f1"] for row in rows),
            }
        )
    result = {
        "protocol": "official-fold-cross-validation",
        "model": "logmel-small-cnn",
        "torch_version": torch.__version__,
        "device": str(device),
        "records": len(records),
        "classes": len(class_names),
        "folds": fold_results,
        "aggregate": aggregate,
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "cross_validation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def load_cnn_artifact(path: Path) -> Tuple[LogMelCNN, dict]:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    model = LogMelCNN(
        len(artifact["class_names"]), artifact["channels"], float(artifact["dropout"])
    )
    model.load_state_dict(artifact["model_state"])
    model.eval()
    return model, artifact


def predict_cnn_file(path: Path, artifact_path: Path) -> List[dict]:
    model, artifact = load_cnn_artifact(artifact_path)
    samples, _ = read_wav_mono(path, int(artifact["sample_rate"]))
    feature = extract_logmel(samples, int(artifact["sample_rate"]), artifact["feature_settings"])
    dummy_label = np.zeros(1, dtype=np.int64)
    _, probabilities, _ = _predict(
        model,
        feature[None, :, :],
        dummy_label,
        tuple(artifact["normalization"]),
        1,
        torch.device("cpu"),
    )
    ranked = np.argsort(probabilities[0])[::-1]
    return [
        {
            "label": artifact["class_names"][int(index)],
            "probability": float(probabilities[0, index]),
        }
        for index in ranked
    ]
