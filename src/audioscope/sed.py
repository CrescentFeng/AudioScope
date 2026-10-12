from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .audio import read_wav_mono
from .features import extract_logmel
from .manifest import AudioRecord
from .sed_data import (
    SoundscapeRecord,
    build_sed_tensors,
    count_polyphony,
    generate_esc10_soundscapes,
    read_soundscape_manifest,
)
from .sed_metrics import evaluate_sed, probabilities_to_events


class CRNN(nn.Module):
    def __init__(
        self,
        n_mels: int,
        num_classes: int,
        channels: Sequence[int],
        gru_hidden_size: int,
        dropout: float,
        time_pool_factors: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if time_pool_factors is None:
            time_pool_factors = [1] * len(channels)
        if len(time_pool_factors) != len(channels):
            raise ValueError("time_pool_factors must match channels")
        blocks = []
        input_channels = 1
        for output_channels, time_pool in zip(channels, time_pool_factors):
            blocks.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(2, int(time_pool))),
                    nn.Dropout2d(dropout * 0.5),
                ]
            )
            input_channels = output_channels
        pooled_mels = n_mels // (2 ** len(channels))
        if pooled_mels < 1:
            raise ValueError("n_mels is too small for the requested convolutional blocks")
        self.encoder = nn.Sequential(*blocks)
        self.recurrent = nn.GRU(
            input_size=input_channels * pooled_mels,
            hidden_size=gru_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(gru_hidden_size * 2, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_frames = inputs.shape[-1]
        encoded = self.encoder(inputs)
        sequence = encoded.permute(0, 3, 1, 2).flatten(2)
        recurrent, _ = self.recurrent(sequence)
        logits = self.head(self.dropout(recurrent))
        if logits.shape[1] != input_frames:
            logits = F.interpolate(
                logits.transpose(1, 2), size=input_frames, mode="linear", align_corners=False
            ).transpose(1, 2)
        return logits


class _ConformerFeedForward(nn.Module):
    def __init__(self, dimension: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class _ConformerBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        feedforward: int,
        convolution_kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if convolution_kernel % 2 == 0:
            raise ValueError("convolution_kernel must be odd")
        self.feedforward_one = _ConformerFeedForward(dimension, feedforward, dropout)
        self.attention_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.convolution_norm = nn.LayerNorm(dimension)
        self.convolution = nn.Sequential(
            nn.Conv1d(dimension, dimension * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(
                dimension,
                dimension,
                kernel_size=convolution_kernel,
                padding=convolution_kernel // 2,
                groups=dimension,
            ),
            nn.BatchNorm1d(dimension),
            nn.SiLU(),
            nn.Conv1d(dimension, dimension, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.feedforward_two = _ConformerFeedForward(dimension, feedforward, dropout)
        self.output_norm = nn.LayerNorm(dimension)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs + 0.5 * self.feedforward_one(inputs)
        normalised = self.attention_norm(outputs)
        attended, _ = self.attention(
            normalised, normalised, normalised, need_weights=False
        )
        outputs = outputs + self.attention_dropout(attended)
        convolved = self.convolution(self.convolution_norm(outputs).transpose(1, 2))
        outputs = outputs + convolved.transpose(1, 2)
        outputs = outputs + 0.5 * self.feedforward_two(outputs)
        return self.output_norm(outputs)


class LiteConformer(nn.Module):
    def __init__(
        self,
        n_mels: int,
        num_classes: int,
        channels: Sequence[int],
        dimension: int,
        heads: int,
        layers: int,
        feedforward: int,
        convolution_kernel: int,
        dropout: float,
        time_pool_factors: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if time_pool_factors is None:
            time_pool_factors = [1] * len(channels)
        if len(time_pool_factors) != len(channels):
            raise ValueError("time_pool_factors must match channels")
        blocks = []
        input_channels = 1
        for output_channels, time_pool in zip(channels, time_pool_factors):
            blocks.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(2, int(time_pool))),
                    nn.Dropout2d(dropout * 0.5),
                ]
            )
            input_channels = output_channels
        pooled_mels = n_mels // (2 ** len(channels))
        if pooled_mels < 1:
            raise ValueError("n_mels is too small for the requested convolutional blocks")
        self.encoder = nn.Sequential(*blocks)
        self.input_projection = nn.Linear(input_channels * pooled_mels, dimension)
        self.blocks = nn.ModuleList(
            [
                _ConformerBlock(
                    dimension,
                    heads,
                    feedforward,
                    convolution_kernel,
                    dropout,
                )
                for _ in range(layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dimension, num_classes)

    @staticmethod
    def _position_encoding(frames: int, dimension: int, device: torch.device) -> torch.Tensor:
        position = torch.arange(frames, device=device, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
            * (-np.log(10000.0) / dimension)
        )
        encoding = torch.zeros(frames, dimension, device=device)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
        return encoding

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_frames = inputs.shape[-1]
        encoded = self.encoder(inputs).permute(0, 3, 1, 2).flatten(2)
        outputs = self.input_projection(encoded)
        outputs = outputs + self._position_encoding(
            outputs.shape[1], outputs.shape[2], outputs.device
        ).unsqueeze(0)
        for block in self.blocks:
            outputs = block(outputs)
        logits = self.head(self.dropout(outputs))
        if logits.shape[1] != input_frames:
            logits = F.interpolate(
                logits.transpose(1, 2), size=input_frames, mode="linear", align_corners=False
            ).transpose(1, 2)
        return logits


def build_sed_model(
    feature_settings: Mapping[str, Any],
    model_settings: Mapping[str, Any],
    num_classes: int,
) -> nn.Module:
    family = str(model_settings.get("family", "crnn")).lower()
    common = {
        "n_mels": int(feature_settings["n_mels"]),
        "num_classes": num_classes,
        "channels": [int(value) for value in model_settings["channels"]],
        "dropout": float(model_settings["dropout"]),
        "time_pool_factors": [
            int(value)
            for value in model_settings.get(
                "time_pool_factors", [1] * len(model_settings["channels"])
            )
        ],
    }
    if family == "crnn":
        return CRNN(
            **common,
            gru_hidden_size=int(model_settings["gru_hidden_size"]),
        )
    if family == "conformer":
        return LiteConformer(
            **common,
            dimension=int(model_settings["conformer_dimension"]),
            heads=int(model_settings["conformer_heads"]),
            layers=int(model_settings["conformer_layers"]),
            feedforward=int(model_settings["conformer_feedforward"]),
            convolution_kernel=int(model_settings["conformer_convolution_kernel"]),
        )
    raise ValueError(f"Unsupported SED model family: {family}")


class SEDDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        normalization: Tuple[float, float],
        frequency_mask: int = 0,
        time_mask: int = 0,
        loss_weights: np.ndarray | None = None,
    ) -> None:
        self.features = features
        self.targets = targets
        self.mean, self.std = normalization
        self.frequency_mask = frequency_mask
        self.time_mask = time_mask
        self.loss_weights = loss_weights

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        feature = torch.from_numpy(self.features[index].copy()).unsqueeze(0)
        feature = (feature - self.mean) / self.std
        target = torch.from_numpy(self.targets[index].astype(np.float32, copy=True))
        if self.frequency_mask > 0:
            width = int(torch.randint(0, self.frequency_mask + 1, (1,)).item())
            if width:
                start = int(torch.randint(0, feature.shape[1] - width + 1, (1,)).item())
                feature[:, start : start + width, :] = 0.0
        if self.time_mask > 0:
            width = int(torch.randint(0, self.time_mask + 1, (1,)).item())
            if width:
                start = int(torch.randint(0, feature.shape[2] - width + 1, (1,)).item())
                feature[:, :, start : start + width] = 0.0
        if self.loss_weights is None:
            loss_weight = torch.ones_like(target)
        else:
            loss_weight = torch.from_numpy(
                self.loss_weights[index].astype(np.float32, copy=True)
            )
        return feature, target, loss_weight


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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


def _predict(
    model: nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    normalization: Tuple[float, float],
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    dataset = SEDDataset(features, targets, normalization)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities = []
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for batch, _, _ in loader:
            probabilities.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    return np.concatenate(probabilities).astype(np.float32), time.perf_counter() - started


def _estimates_from_probabilities(
    records: Sequence[SoundscapeRecord],
    probabilities: np.ndarray,
    class_names: Sequence[str],
    threshold: float | Mapping[str, float] | Sequence[float],
    hop_s: float,
    postprocess: Mapping[str, Any],
) -> dict:
    return {
        record.scene_id: probabilities_to_events(
            scene_probabilities,
            class_names,
            threshold,
            hop_s,
            record.duration_s,
            smooth_frames=int(postprocess.get("smooth_frames", 1)),
            merge_gap_s=float(postprocess.get("merge_gap_s", 0.2)),
            minimum_duration_s=float(postprocess.get("minimum_duration_s", 0.1)),
        )
        for record, scene_probabilities in zip(records, probabilities)
    }


def _evaluate_probabilities(
    records: Sequence[SoundscapeRecord],
    probabilities: np.ndarray,
    class_names: Sequence[str],
    threshold: float | Mapping[str, float] | Sequence[float],
    hop_s: float,
    postprocess: Mapping[str, Any],
    metrics_settings: Mapping[str, Any],
) -> Tuple[dict, dict]:
    estimates = _estimates_from_probabilities(
        records, probabilities, class_names, threshold, hop_s, postprocess
    )
    metrics = evaluate_sed(
        records,
        estimates,
        class_names,
        metrics_settings["event"],
        float(metrics_settings["segment_resolution_s"]),
    )
    return metrics, estimates


def _select_threshold(
    records: Sequence[SoundscapeRecord],
    probabilities: np.ndarray,
    class_names: Sequence[str],
    thresholds: Sequence[float],
    hop_s: float,
    postprocess: Mapping[str, Any],
    metrics_settings: Mapping[str, Any],
) -> Tuple[float, List[dict]]:
    rows = []
    for threshold in thresholds:
        metrics, _ = _evaluate_probabilities(
            records,
            probabilities,
            class_names,
            float(threshold),
            hop_s,
            postprocess,
            metrics_settings,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "event_f1": metrics["event_based"]["micro"]["f1"],
                "segment_f1": metrics["segment_based"]["micro"]["f1"],
            }
        )
    best = max(rows, key=lambda row: (row["event_f1"], row["segment_f1"], -row["threshold"]))
    return float(best["threshold"]), rows


def _select_class_thresholds(
    records: Sequence[SoundscapeRecord],
    probabilities: np.ndarray,
    class_names: Sequence[str],
    thresholds: Sequence[float],
    hop_s: float,
    postprocess: Mapping[str, Any],
    metrics_settings: Mapping[str, Any],
) -> Tuple[dict, List[dict]]:
    rows = []
    candidates: Dict[str, List[dict]] = {label: [] for label in class_names}
    for threshold in thresholds:
        metrics, _ = _evaluate_probabilities(
            records,
            probabilities,
            class_names,
            float(threshold),
            hop_s,
            postprocess,
            metrics_settings,
        )
        for label in class_names:
            event_row = metrics["event_based"]["per_class"][label]
            segment_row = metrics["segment_based"]["per_class"][label]
            row = {
                "class": label,
                "threshold": float(threshold),
                "event_f1": event_row["f1"],
                "segment_f1": segment_row["f1"],
            }
            rows.append(row)
            candidates[label].append(row)
    selected = {
        label: float(
            max(
                candidates[label],
                key=lambda row: (
                    row["event_f1"],
                    row["segment_f1"],
                    row["threshold"],
                ),
            )["threshold"]
        )
        for label in class_names
    }
    return selected, rows


def _valid_frame_mask(
    records: Sequence[SoundscapeRecord], target_frames: int, hop_s: float
) -> np.ndarray:
    frame_times = (np.arange(target_frames, dtype=np.float32) + 0.5) * hop_s
    return np.stack([frame_times < record.duration_s for record in records])


def _loss_weights(
    targets: np.ndarray,
    valid_frames: np.ndarray | None,
    hop_s: float,
    settings: Mapping[str, Any],
) -> np.ndarray:
    weights = np.ones(targets.shape, dtype=np.float32)
    if valid_frames is not None:
        if valid_frames.shape != targets.shape[:2]:
            raise ValueError("valid_frames must have shape [examples, frames]")
        weights *= valid_frames[:, :, None]
    if not settings.get("enabled", False):
        return weights
    short_frames = max(1, int(round(float(settings["maximum_duration_s"]) / hop_s)))
    boundary_frames = max(0, int(round(float(settings.get("boundary_s", 0.0)) / hop_s)))
    short_weight = float(settings.get("positive_weight", 1.0))
    boundary_weight = float(settings.get("boundary_weight", 1.0))
    for example in range(targets.shape[0]):
        for class_index in range(targets.shape[2]):
            activity = targets[example, :, class_index].astype(bool)
            padded = np.pad(activity.astype(np.int8), (1, 1))
            starts = np.flatnonzero(np.diff(padded) == 1)
            ends = np.flatnonzero(np.diff(padded) == -1)
            for start, end in zip(starts, ends):
                if end - start <= short_frames:
                    weights[example, start:end, class_index] = np.maximum(
                        weights[example, start:end, class_index], short_weight
                    )
                if boundary_frames:
                    left = max(0, start - boundary_frames)
                    right = min(targets.shape[1], end + boundary_frames)
                    weights[example, left : start + boundary_frames, class_index] = np.maximum(
                        weights[example, left : start + boundary_frames, class_index],
                        boundary_weight,
                    )
                    weights[example, max(start, end - boundary_frames) : right, class_index] = (
                        np.maximum(
                            weights[
                                example,
                                max(start, end - boundary_frames) : right,
                                class_index,
                            ],
                            boundary_weight,
                        )
                    )
    if valid_frames is not None:
        weights *= valid_frames[:, :, None]
    return weights


def _train(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    class_names: Sequence[str],
    feature_settings: Mapping[str, Any],
    model_settings: Mapping[str, Any],
    seed: int,
    device: torch.device,
    train_valid_frames: np.ndarray | None = None,
    validation_valid_frames: np.ndarray | None = None,
) -> Tuple[nn.Module, Tuple[float, float], dict]:
    _seed_everything(seed)
    torch.set_num_threads(int(model_settings.get("torch_threads", 8)))
    normalization = (
        float(train_features.mean()),
        float(train_features.std() + 1e-6),
    )
    augmentation = model_settings.get("augmentation", {})
    hop_s = float(feature_settings["hop_ms"]) / 1000.0
    short_event_settings = model_settings.get("short_event_optimization", {})
    train_dataset = SEDDataset(
        train_features,
        train_targets,
        normalization,
        int(augmentation.get("frequency_mask", 0)),
        int(augmentation.get("time_mask", 0)),
        _loss_weights(
            train_targets, train_valid_frames, hop_s, short_event_settings
        ),
    )
    validation_dataset = SEDDataset(
        validation_features,
        validation_targets,
        normalization,
        loss_weights=_loss_weights(
            validation_targets, validation_valid_frames, hop_s, {"enabled": False}
        ),
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if model_settings.get("balanced_sampling", False):
        presence = train_targets.max(axis=1) > 0
        frequency = presence.sum(axis=0).astype(np.float64)
        rarity = np.sqrt(len(presence) / np.maximum(frequency, 1.0))
        weights = np.maximum(1.0, (presence * rarity[None, :]).max(axis=1))
        weights = np.minimum(weights, 3.0)
        sampler = WeightedRandomSampler(
            weights, len(train_dataset), replacement=True, generator=generator
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(model_settings["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(model_settings["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    model = build_sed_model(feature_settings, model_settings, len(class_names)).to(device)
    positives = train_targets.sum(axis=(0, 1)).astype(np.float64)
    negatives = train_targets.shape[0] * train_targets.shape[1] - positives
    pos_weight = np.clip(
        negatives / np.maximum(positives, 1.0),
        1.0,
        float(model_settings.get("max_pos_weight", 20.0)),
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device),
        reduction="none",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_settings["learning_rate"]),
        weight_decay=float(model_settings["weight_decay"]),
    )
    history = []
    best_state = None
    best_validation_loss = float("inf")
    patience = int(model_settings.get("early_stopping_patience", 8))
    epochs_without_improvement = 0
    started = time.perf_counter()
    for epoch in range(int(model_settings["epochs"])):
        model.train()
        train_loss = 0.0
        train_frames = 0
        for features, targets, loss_weights in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            loss_weights = loss_weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            raw_loss = criterion(logits, targets)
            loss = (raw_loss * loss_weights).sum() / loss_weights.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(features)
            train_frames += len(features)
        model.eval()
        validation_loss = 0.0
        validation_frames = 0
        with torch.inference_mode():
            for features, targets, loss_weights in validation_loader:
                features = features.to(device)
                targets = targets.to(device)
                loss_weights = loss_weights.to(device)
                raw_loss = criterion(model(features), targets)
                loss = (raw_loss * loss_weights).sum() / loss_weights.sum().clamp_min(1.0)
                validation_loss += float(loss.item()) * len(features)
                validation_frames += len(features)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss / train_frames,
            "validation_loss": validation_loss / validation_frames,
        }
        history.append(row)
        print(
            f"SED epoch {epoch + 1}: train={row['train_loss']:.4f} "
            f"validation={row['validation_loss']:.4f}",
            flush=True,
        )
        if row["validation_loss"] < best_validation_loss - 1e-5:
            best_validation_loss = row["validation_loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is None:
        raise RuntimeError("No CRNN checkpoint was produced")
    model.load_state_dict(best_state)
    return model, normalization, {
        "epochs_completed": len(history),
        "best_validation_loss": best_validation_loss,
        "training_seconds": time.perf_counter() - started,
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "positive_weights": dict(zip(class_names, pos_weight.tolist())),
        "short_event_optimization": dict(short_event_settings),
        "history": history,
    }


def _write_predictions(path: Path, estimates: Mapping[str, Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for scene_id, events in estimates.items():
            handle.write(
                json.dumps(
                    {
                        "scene_id": scene_id,
                        "events": [
                            {
                                "label": event.label,
                                "onset_s": event.onset_s,
                                "offset_s": event.offset_s,
                            }
                            for event in events
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _write_metric_tables(report_root: Path, metrics: Mapping[str, Any]) -> None:
    with (report_root / "per_class_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "class",
                "event_precision",
                "event_recall",
                "event_f1",
                "segment_precision",
                "segment_recall",
                "segment_f1",
            ]
        )
        for label, event_row in metrics["event_based"]["per_class"].items():
            segment_row = metrics["segment_based"]["per_class"][label]
            writer.writerow(
                [
                    label,
                    event_row["precision"],
                    event_row["recall"],
                    event_row["f1"],
                    segment_row["precision"],
                    segment_row["recall"],
                    segment_row["f1"],
                ]
            )


def run_sed_experiment(
    source_records: Sequence[AudioRecord],
    source_root: Path,
    soundscape_root: Path,
    generation_settings: Mapping[str, Any],
    feature_settings: Mapping[str, Any],
    model_settings: Mapping[str, Any],
    postprocess_settings: Mapping[str, Any],
    metrics_settings: Mapping[str, Any],
    seed: int,
    artifact_root: Path,
    report_root: Path,
    cache_root: Path,
) -> dict:
    generation = generate_esc10_soundscapes(
        source_records, source_root, soundscape_root, generation_settings, seed
    )
    records = read_soundscape_manifest(soundscape_root / "manifest.jsonl")
    class_names = list(generation["classes"])
    split_records = {
        split: [record for record in records if record.split == split]
        for split in ("train", "validation", "test")
    }
    tensors = {}
    cache_metadata = {}
    for split, items in split_records.items():
        features, targets, metadata = build_sed_tensors(
            items,
            soundscape_root,
            class_names,
            feature_settings,
            cache_root / f"{split}.npz",
        )
        tensors[split] = (features, targets)
        cache_metadata[split] = metadata
    device = _device(str(model_settings.get("device", "auto")))
    model, normalization, training = _train(
        *tensors["train"],
        *tensors["validation"],
        class_names,
        feature_settings,
        model_settings,
        seed,
        device,
    )
    validation_probabilities, validation_inference_seconds = _predict(
        model,
        *tensors["validation"],
        normalization,
        int(model_settings["batch_size"]),
        device,
    )
    hop_s = float(feature_settings["hop_ms"]) / 1000.0
    threshold, threshold_rows = _select_threshold(
        split_records["validation"],
        validation_probabilities,
        class_names,
        [float(value) for value in postprocess_settings["threshold_grid"]],
        hop_s,
        postprocess_settings,
        metrics_settings,
    )
    validation_metrics, validation_estimates = _evaluate_probabilities(
        split_records["validation"],
        validation_probabilities,
        class_names,
        threshold,
        hop_s,
        postprocess_settings,
        metrics_settings,
    )
    test_probabilities, test_inference_seconds = _predict(
        model,
        *tensors["test"],
        normalization,
        int(model_settings["batch_size"]),
        device,
    )
    test_metrics, test_estimates = _evaluate_probabilities(
        split_records["test"],
        test_probabilities,
        class_names,
        threshold,
        hop_s,
        postprocess_settings,
        metrics_settings,
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.cpu().state_dict(),
            "class_names": class_names,
            "normalization": normalization,
            "feature_settings": dict(feature_settings),
            "model_settings": dict(model_settings),
            "postprocess_settings": dict(postprocess_settings),
            "threshold": threshold,
            "sample_rate": int(generation_settings["sample_rate"]),
            "seed": seed,
        },
        artifact_root / "crnn.pt",
    )
    report_root.mkdir(parents=True, exist_ok=True)
    _write_predictions(report_root / "validation_predictions.jsonl", validation_estimates)
    _write_predictions(report_root / "test_predictions.jsonl", test_estimates)
    with (report_root / "threshold_sweep.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["threshold", "event_f1", "segment_f1"])
        writer.writeheader()
        writer.writerows(threshold_rows)
    _write_metric_tables(report_root, test_metrics)
    dataset_analysis = {
        split: {
            "scenes": len(items),
            "events": sum(len(record.events) for record in items),
            "maximum_polyphony": max(count_polyphony(record) for record in items),
        }
        for split, items in split_records.items()
    }
    result = {
        "protocol": "source-disjoint-polyphonic-esc10-strong-label-sed",
        "generation": generation,
        "dataset_analysis": dataset_analysis,
        "feature_cache": cache_metadata,
        "model": "logmel-crnn",
        "training": training,
        "selected_threshold": threshold,
        "threshold_selection_split": "validation",
        "validation": {
            "metrics": validation_metrics,
            "inference_seconds_logmel_only": validation_inference_seconds,
        },
        "test": {
            "metrics": test_metrics,
            "inference_seconds_logmel_only": test_inference_seconds,
        },
    }
    (artifact_root / "metadata.json").write_text(
        json.dumps(
            {
                "class_names": class_names,
                "normalization": normalization,
                "threshold": threshold,
                "training": training,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (report_root / "sed_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def predict_sed_file(
    audio_path: Path, artifact_path: Path, threshold_profile: str = "overall"
) -> List[dict]:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    model_settings = artifact["model_settings"]
    feature_settings = artifact["feature_settings"]
    class_names = artifact["class_names"]
    model = build_sed_model(feature_settings, model_settings, len(class_names))
    model.load_state_dict(artifact["model_state"])
    model.eval()
    sample_rate = int(artifact["sample_rate"])
    samples, _ = read_wav_mono(audio_path, sample_rate)
    feature = extract_logmel(samples, sample_rate, feature_settings)
    target_frames = int(feature_settings["target_frames"])
    dummy_targets = np.zeros((1, target_frames, len(class_names)), dtype=np.uint8)
    probabilities, _ = _predict(
        model,
        feature[None, :, :],
        dummy_targets,
        tuple(artifact["normalization"]),
        batch_size=1,
        device=torch.device("cpu"),
    )
    hop_s = float(feature_settings["hop_ms"]) / 1000.0
    duration_s = min(len(samples) / sample_rate, target_frames * hop_s)
    if threshold_profile == "overall":
        threshold = artifact["threshold"]
    elif threshold_profile == "class_balanced" and "class_thresholds" in artifact:
        threshold = artifact["class_thresholds"]
    elif threshold_profile == "short_event" and "short_focused_threshold" in artifact:
        threshold = artifact["short_focused_threshold"]
    else:
        raise ValueError(
            f"Threshold profile {threshold_profile!r} is unavailable in {artifact_path}"
        )
    events = probabilities_to_events(
        probabilities[0],
        class_names,
        threshold,
        hop_s,
        duration_s,
        smooth_frames=int(artifact["postprocess_settings"].get("smooth_frames", 1)),
        merge_gap_s=float(artifact["postprocess_settings"].get("merge_gap_s", 0.2)),
        minimum_duration_s=float(
            artifact["postprocess_settings"].get("minimum_duration_s", 0.1)
        ),
    )
    return [
        {"label": event.label, "onset_s": event.onset_s, "offset_s": event.offset_s}
        for event in events
    ]
