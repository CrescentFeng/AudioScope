from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .desed import DESED_CLASSES, prepare_desed_real
from .sed import (
    _device,
    _evaluate_probabilities,
    _predict,
    _select_class_thresholds,
    _select_threshold,
    _train,
    _valid_frame_mask,
    _write_metric_tables,
    _write_predictions,
    build_sed_model,
)
from .sed_data import build_sed_tensors, read_soundscape_manifest
from .sed_metrics import duration_binned_event_metrics


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _comparison_row(result: Mapping[str, Any]) -> dict:
    training = result["training"]
    test_global = result["test_global_metrics"]
    test_metrics = result["test_metrics"]
    duration_metrics = result["test_duration_metrics"]
    global_duration_metrics = result["test_global_duration_metrics"]
    return {
        "variant": result["name"],
        "family": result["family"],
        "parameters": training["parameters"],
        "validation_loss": training["best_validation_loss"],
        "global_threshold": result["global_threshold"],
        "short_focused_threshold": result["short_focused_threshold"],
        "test_global_event_f1": test_global["event_based"]["micro"]["f1"],
        "test_global_macro_event_f1": test_global["event_based"]["macro_f1"],
        "test_class_threshold_event_f1": test_metrics["event_based"]["micro"]["f1"],
        "test_class_threshold_macro_event_f1": test_metrics["event_based"]["macro_f1"],
        "test_global_segment_f1": test_global["segment_based"]["micro"]["f1"],
        "test_class_threshold_segment_f1": test_metrics["segment_based"]["micro"]["f1"],
        "test_global_short_event_f1": global_duration_metrics["short"]["micro"]["f1"],
        "test_class_threshold_short_event_f1": duration_metrics["short"]["micro"]["f1"],
        "test_short_focused_short_event_f1": result["test_short_focused_duration_metrics"][
            "short"
        ]["micro"]["f1"],
        "test_global_long_event_f1": global_duration_metrics["long"]["micro"]["f1"],
        "test_class_threshold_long_event_f1": duration_metrics["long"]["micro"]["f1"],
        "training_seconds": training["training_seconds"],
        "test_inference_seconds": result["test_inference_seconds_logmel_only"],
    }


def _select_short_event_threshold(
    records,
    probabilities,
    class_names,
    thresholds,
    hop_s,
    postprocess_settings,
    metrics_settings,
    short_event_max_s,
):
    rows = []
    for threshold in thresholds:
        metrics, estimates = _evaluate_probabilities(
            records,
            probabilities,
            class_names,
            threshold,
            hop_s,
            postprocess_settings,
            metrics_settings,
        )
        duration_metrics = duration_binned_event_metrics(
            records,
            estimates,
            class_names,
            metrics_settings["event"],
            short_event_max_s,
        )
        rows.append(
            {
                "threshold": threshold,
                "short_event_f1": duration_metrics["short"]["micro"]["f1"],
                "overall_event_f1": metrics["event_based"]["micro"]["f1"],
                "segment_f1": metrics["segment_based"]["micro"]["f1"],
            }
        )
    best = max(
        rows,
        key=lambda row: (
            row["short_event_f1"],
            row["overall_event_f1"],
            row["threshold"],
        ),
    )
    return float(best["threshold"]), rows


def run_desed_enhancement(
    dataset_settings: Mapping[str, Any],
    feature_settings: Mapping[str, Any],
    base_model_settings: Mapping[str, Any],
    variants: list[Mapping[str, Any]],
    postprocess_settings: Mapping[str, Any],
    metrics_settings: Mapping[str, Any],
    seed: int,
    shard_root: Path,
    dataset_root: Path,
    cache_root: Path,
    artifact_root: Path,
    report_root: Path,
) -> dict:
    dataset_metadata = prepare_desed_real(shard_root, dataset_root, dataset_settings)
    records = read_soundscape_manifest(dataset_root / "manifest.jsonl")
    split_records = {
        split: [record for record in records if record.split == split]
        for split in ("train", "validation", "test")
    }
    class_names = list(DESED_CLASSES)
    tensors = {}
    cache_metadata = {}
    for split, items in split_records.items():
        features, targets, metadata = build_sed_tensors(
            items,
            dataset_root,
            class_names,
            feature_settings,
            cache_root / f"{split}.npz",
        )
        tensors[split] = (features, targets)
        cache_metadata[split] = metadata

    target_frames = int(feature_settings["target_frames"])
    hop_s = float(feature_settings["hop_ms"]) / 1000.0
    valid_frames = {
        split: _valid_frame_mask(items, target_frames, hop_s)
        for split, items in split_records.items()
    }
    device = _device(str(base_model_settings.get("device", "auto")))
    thresholds = [float(value) for value in postprocess_settings["threshold_grid"]]
    short_event_max_s = float(metrics_settings.get("short_event_max_s", 1.0))
    artifact_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    variant_results = []
    comparison_rows = []
    for variant in variants:
        name = str(variant["name"])
        model_settings = dict(base_model_settings)
        model_settings.update({key: value for key, value in variant.items() if key != "name"})
        variant_report_root = report_root / name
        variant_report_root.mkdir(parents=True, exist_ok=True)
        summary_path = variant_report_root / "summary.json"
        artifact_path = artifact_root / f"{name}.pt"
        if summary_path.is_file() and artifact_path.is_file():
            result = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                result.get("threshold_grid") == thresholds
                and result.get("report_schema_version") == 4
            ):
                print(f"\n=== Reusing completed DESED variant: {name} ===", flush=True)
                variant_results.append(result)
                comparison_rows.append(_comparison_row(result))
                continue
            print(f"\n=== Retuning saved DESED variant: {name} ===", flush=True)
            checkpoint = torch.load(artifact_path, map_location="cpu", weights_only=True)
            model_settings = checkpoint["model_settings"]
            model = build_sed_model(feature_settings, model_settings, len(class_names))
            model.load_state_dict(checkpoint["model_state"])
            model = model.to(device)
            normalization = tuple(checkpoint["normalization"])
            training = result["training"]
        else:
            print(f"\n=== DESED enhancement variant: {name} ===", flush=True)
            model, normalization, training = _train(
                *tensors["train"],
                *tensors["validation"],
                class_names,
                feature_settings,
                model_settings,
                seed,
                device,
                train_valid_frames=valid_frames["train"],
                validation_valid_frames=valid_frames["validation"],
            )
        validation_probabilities, validation_seconds = _predict(
            model,
            *tensors["validation"],
            normalization,
            int(model_settings["batch_size"]),
            device,
        )
        global_threshold, global_rows = _select_threshold(
            split_records["validation"],
            validation_probabilities,
            class_names,
            thresholds,
            hop_s,
            postprocess_settings,
            metrics_settings,
        )
        class_thresholds, class_rows = _select_class_thresholds(
            split_records["validation"],
            validation_probabilities,
            class_names,
            thresholds,
            hop_s,
            postprocess_settings,
            metrics_settings,
        )
        short_focused_threshold, short_threshold_rows = _select_short_event_threshold(
            split_records["validation"],
            validation_probabilities,
            class_names,
            thresholds,
            hop_s,
            postprocess_settings,
            metrics_settings,
            short_event_max_s,
        )
        validation_global_metrics, _ = _evaluate_probabilities(
            split_records["validation"],
            validation_probabilities,
            class_names,
            global_threshold,
            hop_s,
            postprocess_settings,
            metrics_settings,
        )
        validation_metrics, validation_estimates = _evaluate_probabilities(
            split_records["validation"],
            validation_probabilities,
            class_names,
            class_thresholds,
            hop_s,
            postprocess_settings,
            metrics_settings,
        )
        test_probabilities, test_seconds = _predict(
            model,
            *tensors["test"],
            normalization,
            int(model_settings["batch_size"]),
            device,
        )
        test_global_metrics, test_global_estimates = _evaluate_probabilities(
            split_records["test"],
            test_probabilities,
            class_names,
            global_threshold,
            hop_s,
            postprocess_settings,
            metrics_settings,
        )
        test_metrics, test_estimates = _evaluate_probabilities(
            split_records["test"],
            test_probabilities,
            class_names,
            class_thresholds,
            hop_s,
            postprocess_settings,
            metrics_settings,
        )
        test_short_focused_metrics, test_short_focused_estimates = _evaluate_probabilities(
            split_records["test"],
            test_probabilities,
            class_names,
            short_focused_threshold,
            hop_s,
            postprocess_settings,
            metrics_settings,
        )
        duration_metrics = duration_binned_event_metrics(
            split_records["test"],
            test_estimates,
            class_names,
            metrics_settings["event"],
            short_event_max_s,
        )
        global_duration_metrics = duration_binned_event_metrics(
            split_records["test"],
            test_global_estimates,
            class_names,
            metrics_settings["event"],
            short_event_max_s,
        )
        short_focused_duration_metrics = duration_binned_event_metrics(
            split_records["test"],
            test_short_focused_estimates,
            class_names,
            metrics_settings["event"],
            short_event_max_s,
        )

        checkpoint = {
            "model_state": model.cpu().state_dict(),
            "variant": name,
            "class_names": class_names,
            "normalization": normalization,
            "feature_settings": dict(feature_settings),
            "model_settings": model_settings,
            "postprocess_settings": dict(postprocess_settings),
            "threshold": global_threshold,
            "global_threshold": global_threshold,
            "class_thresholds": class_thresholds,
            "short_focused_threshold": short_focused_threshold,
            "threshold_profiles": {
                "overall": global_threshold,
                "class_balanced": class_thresholds,
                "short_event": short_focused_threshold,
            },
            "sample_rate": int(dataset_settings["sample_rate"]),
            "seed": seed,
        }
        torch.save(checkpoint, artifact_path)
        artifact_metadata = {
            "variant": name,
            "family": model_settings["family"],
            "class_names": class_names,
            "class_thresholds": class_thresholds,
            "global_threshold": global_threshold,
            "short_focused_threshold": short_focused_threshold,
            "threshold_profiles": {
                "overall": global_threshold,
                "class_balanced": class_thresholds,
                "short_event": short_focused_threshold,
            },
            "training": training,
        }
        (artifact_root / f"{name}_metadata.json").write_text(
            json.dumps(artifact_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        _write_predictions(
            variant_report_root / "validation_predictions.jsonl", validation_estimates
        )
        _write_predictions(variant_report_root / "test_predictions.jsonl", test_estimates)
        _write_metric_tables(variant_report_root, test_metrics)
        _write_csv(
            variant_report_root / "global_threshold_sweep.csv",
            global_rows,
            ["threshold", "event_f1", "segment_f1"],
        )
        _write_csv(
            variant_report_root / "per_class_threshold_sweep.csv",
            class_rows,
            ["class", "threshold", "event_f1", "segment_f1"],
        )
        _write_csv(
            variant_report_root / "short_event_threshold_sweep.csv",
            short_threshold_rows,
            ["threshold", "short_event_f1", "overall_event_f1", "segment_f1"],
        )
        _write_csv(
            variant_report_root / "selected_class_thresholds.csv",
            [
                {"class": label, "threshold": class_thresholds[label]}
                for label in class_names
            ],
            ["class", "threshold"],
        )

        result = {
            "report_schema_version": 4,
            "name": name,
            "family": model_settings["family"],
            "threshold_grid": thresholds,
            "model_settings": model_settings,
            "training": training,
            "global_threshold": global_threshold,
            "class_thresholds": class_thresholds,
            "short_focused_threshold": short_focused_threshold,
            "validation_global_metrics": validation_global_metrics,
            "validation_metrics": validation_metrics,
            "validation_inference_seconds_logmel_only": validation_seconds,
            "test_global_metrics": test_global_metrics,
            "test_metrics": test_metrics,
            "test_short_focused_metrics": test_short_focused_metrics,
            "test_global_duration_metrics": global_duration_metrics,
            "test_duration_metrics": duration_metrics,
            "test_short_focused_duration_metrics": short_focused_duration_metrics,
            "test_inference_seconds_logmel_only": test_seconds,
        }
        summary_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        variant_results.append(result)
        comparison_rows.append(_comparison_row(result))

    _write_csv(
        report_root / "model_comparison.csv",
        comparison_rows,
        list(comparison_rows[0]),
    )
    summary = {
        "protocol": "desed-real-strong-source-disjoint-enhanced-sed",
        "dataset": dataset_metadata,
        "feature_cache": cache_metadata,
        "threshold_selection_split": "validation",
        "test_used_for_selection": False,
        "variants": variant_results,
    }
    (report_root / "enhancement_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
