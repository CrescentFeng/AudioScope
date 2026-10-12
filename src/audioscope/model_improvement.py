"""Fixed-budget, validation-selected paired experiments; no test data during development.

Run development first. A separate held-out evaluation will follow frozen selection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .audio import read_wav_mono
from .features import extract_logmel
from .sed import _evaluate_probabilities, _predict, _train, _valid_frame_mask
from .sed_data import build_sed_tensors, read_soundscape_manifest
from .software_validation import CONDITIONS, transform, write_json
from .workbench import sha256

SEEDS = (2027, 2028, 2029)
VARIANTS = ("paired_baseline", "balanced_domain_augmented")


def run_development(root: Path):
    config = yaml.safe_load((root / "configs/desed_real_enhanced.yaml").read_text())
    output = root / "reports/generated/model-improvement"
    artifacts = root / "artifacts/model-improvement"
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest = root / "data/desed-real-strong-only/manifest.jsonl"
    records = read_soundscape_manifest(manifest)
    train = [r for r in records if r.split == "train"]
    validation = [r for r in records if r.split == "validation"]
    # Never load, transform, train on, select from, or evaluate test audio in development.
    features = config["features"]
    model_settings = dict(config["model"])
    model_settings.update(config["comparison"]["variants"][1])
    model_settings.update(device="cpu", torch_threads=4)
    model_settings.pop("name", None)
    protocol = {
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "manifest_sha256": sha256(manifest),
        "train_clips": len(train),
        "validation_clips": len(validation),
        "test_used": False,
        "model_settings": model_settings,
        "features": features,
        "postprocess": config["postprocess"],
        "conditions": CONDITIONS,
        "selection": "mean across seeds of mean clean/shift validation micro and macro event F1",
        "promotion_guard": "mean clean validation micro F1 must not fall by more than 0.01",
        "augmentation": "one deterministic shifted copy per training clip plus all clean clips",
        "training_seeds": "record-index + run-seed; validation shifts use 8127+record-index",
        "baseline_preserved": True,
    }
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol["signature"] = signature
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing development protocol differs; use a new experiment directory")
    write_json(protocol_path, protocol)
    cache = root / "data/processed/desed-real-strong-only-sed"
    from .desed import DESED_CLASSES

    classes = list(DESED_CLASSES)
    x_train, y_train, _ = build_sed_tensors(
        train, manifest.parent, classes, features, cache / "train.npz"
    )
    x_val, y_val, _ = build_sed_tensors(
        validation, manifest.parent, classes, features, cache / "validation.npz"
    )
    hop_s = features["hop_ms"] / 1000
    train_mask = _valid_frame_mask(train, features["target_frames"], hop_s)
    val_mask = _valid_frame_mask(validation, features["target_frames"], hop_s)
    validation_features = {"clean": x_val}
    for condition in CONDITIONS[1:]:
        validation_features[condition] = np.stack(
            [
                extract_logmel(
                    transform(
                        read_wav_mono(manifest.parent / r.path, 16000)[0],
                        16000,
                        condition,
                        8127 + i,
                    ),
                    16000,
                    features,
                )
                for i, r in enumerate(validation)
            ]
        )
    results = []
    for seed in SEEDS:
        for variant in VARIANTS:
            key = f"{variant}-{seed}"
            summary = output / f"{key}.json"
            checkpoint_path = artifacts / f"{key}.pt"
            if summary.exists() and checkpoint_path.exists():
                prior = json.loads(summary.read_text())
                if prior["protocol_signature"] == signature and prior["sha256"] == sha256(
                    checkpoint_path
                ):
                    results.append(prior)
                    continue
            print(f"Starting {key}", flush=True)
            settings = copy.deepcopy(model_settings)
            settings["balanced_sampling"] = variant != "paired_baseline"
            train_x, train_y, mask = x_train, y_train, train_mask
            if settings["balanced_sampling"]:
                augmented = np.stack(
                    [
                        extract_logmel(
                            transform(
                                read_wav_mono(manifest.parent / r.path, 16000)[0],
                                16000,
                                CONDITIONS[1 + (i + seed) % 6],
                                seed + i,
                            ),
                            16000,
                            features,
                        )
                        for i, r in enumerate(train)
                    ]
                )
                train_x = np.concatenate([x_train, augmented])
                train_y = np.concatenate([y_train, y_train])
                mask = np.concatenate([train_mask, train_mask])
                # Equal optimizer-update budget: twice the examples, half the epochs.
                settings["epochs"] = model_settings["epochs"] // 2
            model, normalization, training = _train(
                train_x,
                train_y,
                x_val,
                y_val,
                classes,
                features,
                settings,
                seed,
                torch.device("cpu"),
                train_valid_frames=mask,
                validation_valid_frames=val_mask,
            )
            probabilities = {
                condition: _predict(
                    model, values, y_val, normalization, settings["batch_size"], torch.device("cpu")
                )[0]
                for condition, values in validation_features.items()
            }
            sweep = []
            for threshold in config["postprocess"]["threshold_grid"]:
                scores = {}
                for condition, probs in probabilities.items():
                    metric, _ = _evaluate_probabilities(
                        validation,
                        probs,
                        classes,
                        threshold,
                        hop_s,
                        config["postprocess"],
                        config["metrics"],
                    )
                    scores[condition] = metric
                value = np.mean(
                    [
                        (m["event_based"]["micro"]["f1"] + m["event_based"]["macro_f1"]) / 2
                        for m in scores.values()
                    ]
                )
                sweep.append({"threshold": threshold, "objective": float(value), "scores": scores})
            best = max(sweep, key=lambda item: (item["objective"], item["threshold"]))
            torch.save(
                {
                    "model_state": model.cpu().state_dict(),
                    "class_names": classes,
                    "normalization": normalization,
                    "feature_settings": features,
                    "model_settings": settings,
                    "postprocess_settings": config["postprocess"],
                    "threshold": best["threshold"],
                    "threshold_profiles": {"overall": best["threshold"]},
                    "sample_rate": 16000,
                    "seed": seed,
                },
                checkpoint_path,
            )
            result = {
                "variant": variant,
                "seed": seed,
                "protocol_signature": signature,
                "sha256": sha256(checkpoint_path),
                "checkpoint": str(checkpoint_path),
                "training": training,
                "selection": best,
                "threshold_sweep": sweep,
            }
            write_json(summary, result)
            results.append(result)
            print(f"Finished {key}, validation objective={best['objective']:.4f}", flush=True)
    aggregates = {}
    for variant in VARIANTS:
        rows = [r for r in results if r["variant"] == variant]
        values = [r["selection"]["objective"] for r in rows]
        aggregates[variant] = {
            "mean_objective": float(np.mean(values)),
            "std_objective": float(np.std(values)),
            "mean_clean_micro": float(
                np.mean(
                    [r["selection"]["scores"]["clean"]["event_based"]["micro"]["f1"] for r in rows]
                )
            ),
        }
    baseline, candidate = [aggregates[v] for v in VARIANTS]
    winner = VARIANTS[
        int(
            candidate["mean_objective"] > baseline["mean_objective"]
            and candidate["mean_clean_micro"] >= baseline["mean_clean_micro"] - 0.01
        )
    ]
    write_json(
        output / "development_summary.json",
        {
            "status": "completed",
            "protocol_signature": signature,
            "aggregates": aggregates,
            "selected_variant": winner,
            "test_used": False,
            "default_model_changed": False,
            "note": "A completed development experiment is not a release acceptance.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run_development(args.root.resolve())
