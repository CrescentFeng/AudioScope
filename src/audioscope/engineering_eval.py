from __future__ import annotations

import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.io import wavfile
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import roc_auc_score, roc_curve

from .audio import add_white_noise, read_wav_mono
from .engineering import SEDDeployment, frame_rms_dbfs, stream_file
from .features import extract_logmel
from .manifest import read_manifest
from .sed import (
    _evaluate_probabilities,
    _select_class_thresholds,
    _select_threshold,
    build_sed_model,
)
from .sed_data import SoundscapeRecord, TimedEvent, read_soundscape_manifest
from .sed_enhanced import _select_short_event_threshold
from .sed_metrics import evaluate_sed


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(value), indent=2, ensure_ascii=False), encoding="utf-8")


def _load_torch_model(artifact: Mapping[str, Any]) -> torch.nn.Module:
    model = build_sed_model(
        artifact["feature_settings"], artifact["model_settings"], len(artifact["class_names"])
    )
    model.load_state_dict(artifact["model_state"])
    model.eval()
    return model


def _torch_logits(
    model: torch.nn.Module,
    features: np.ndarray,
    normalization: Sequence[float],
    batch_size: int = 32,
) -> np.ndarray:
    mean, std = (float(value) for value in normalization)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(
                ((features[start : start + batch_size] - mean) / std)[:, None].astype(np.float32)
            )
            outputs.append(model(batch).numpy())
    return np.concatenate(outputs).astype(np.float32)


def _binary_calibration_metrics(logits: np.ndarray, targets: np.ndarray) -> dict:
    probabilities = expit(np.clip(logits, -40.0, 40.0))
    target = targets.astype(np.float64).reshape(-1)
    probability = probabilities.astype(np.float64).reshape(-1)
    eps = 1e-8
    nll = -np.mean(
        target * np.log(np.maximum(probability, eps))
        + (1.0 - target) * np.log(np.maximum(1.0 - probability, eps))
    )
    brier = np.mean(np.square(probability - target))
    ece = 0.0
    bin_rows = []
    for left in np.linspace(0.0, 0.9, 10):
        right = left + 0.1
        mask = (probability >= left) & (
            probability <= right if right >= 1.0 else probability < right
        )
        if not np.any(mask):
            continue
        confidence = float(probability[mask].mean())
        observed = float(target[mask].mean())
        fraction = float(mask.mean())
        ece += fraction * abs(confidence - observed)
        bin_rows.append(
            {
                "left": left,
                "right": right,
                "count": int(mask.sum()),
                "mean_confidence": confidence,
                "observed_frequency": observed,
            }
        )
    return {"nll": float(nll), "brier": float(brier), "ece_10_bin": ece, "bins": bin_rows}


def calibrate_artifact(
    source_artifact: Path,
    validation_cache: Path,
    validation_manifest: Path,
    output_artifact: Path,
    threshold_grid: Sequence[float],
    metrics_settings: Mapping[str, Any],
) -> dict:
    artifact = torch.load(source_artifact, map_location="cpu", weights_only=True)
    cache = np.load(validation_cache)
    features = cache["features"].astype(np.float32)
    targets = cache["targets"].astype(np.uint8)
    records = [
        record
        for record in read_soundscape_manifest(validation_manifest)
        if record.split == "validation"
    ]
    if len(records) != len(features):
        raise ValueError("validation cache and manifest have different lengths")
    model = _load_torch_model(artifact)
    logits = _torch_logits(model, features, artifact["normalization"])

    def objective(log_temperature: float) -> float:
        return _binary_calibration_metrics(logits / math.exp(float(log_temperature)), targets)[
            "nll"
        ]

    result = minimize_scalar(objective, bounds=(-2.0, 2.0), method="bounded")
    temperature = float(math.exp(result.x))
    before = _binary_calibration_metrics(logits, targets)
    after = _binary_calibration_metrics(logits / temperature, targets)
    accepted = bool(after["nll"] < before["nll"] and after["ece_10_bin"] <= before["ece_10_bin"])
    if not accepted:
        temperature = 1.0
        after = before
    probabilities = expit(np.clip(logits / temperature, -40.0, 40.0)).astype(np.float32)
    hop_s = float(artifact["feature_settings"]["hop_ms"]) / 1000.0
    postprocess = artifact["postprocess_settings"]
    grid = [float(value) for value in threshold_grid]
    overall, overall_rows = _select_threshold(
        records,
        probabilities,
        artifact["class_names"],
        grid,
        hop_s,
        postprocess,
        metrics_settings,
    )
    class_thresholds, class_rows = _select_class_thresholds(
        records,
        probabilities,
        artifact["class_names"],
        grid,
        hop_s,
        postprocess,
        metrics_settings,
    )
    short, short_rows = _select_short_event_threshold(
        records,
        probabilities,
        artifact["class_names"],
        grid,
        hop_s,
        postprocess,
        metrics_settings,
        float(metrics_settings.get("short_event_max_s", 1.0)),
    )
    artifact["threshold"] = overall
    artifact["global_threshold"] = overall
    artifact["class_thresholds"] = class_thresholds
    artifact["short_focused_threshold"] = short
    artifact["threshold_profiles"] = {
        "overall": overall,
        "class_balanced": class_thresholds,
        "short_event": short,
    }
    artifact["calibration"] = _json_ready(
        {
            "method": "scalar_temperature_multilabel_nll",
            "temperature": temperature,
            "accepted": accepted,
            "selection_split": "validation",
            "before": before,
            "after": after,
        }
    )
    output_artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_artifact)
    report = _json_ready(
        {
            "source_artifact": str(source_artifact),
            "output_artifact": str(output_artifact),
            "temperature": temperature,
            "accepted": accepted,
            "before": before,
            "after": after,
            "threshold_profiles": artifact["threshold_profiles"],
            "overall_threshold_sweep": overall_rows,
            "class_threshold_sweep": class_rows,
            "short_event_threshold_sweep": short_rows,
        }
    )
    return report


def calibrate_open_set(
    artifact_path: Path,
    validation_cache: Path,
    esc_manifest: Path,
    esc_root: Path,
    maximum_unknown_clips: int,
    activity_dbfs: float,
    seed: int,
) -> dict:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    model = _load_torch_model(artifact)
    cache = np.load(validation_cache)
    known_features = cache["features"].astype(np.float32)
    known_targets = cache["targets"].astype(bool)
    temperature = float(artifact.get("calibration", {}).get("temperature", 1.0))
    known_probabilities = expit(
        np.clip(
            _torch_logits(model, known_features, artifact["normalization"]) / temperature,
            -40.0,
            40.0,
        )
    )
    known_scores = known_probabilities.max(axis=2)[known_targets.any(axis=2)]

    known_labels = set(artifact["class_names"])
    candidates = [
        record for record in read_manifest(esc_manifest) if record.label not in known_labels
    ]
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(
        len(candidates), size=min(maximum_unknown_clips, len(candidates)), replace=False
    )
    selected = [candidates[int(index)] for index in selected_indices]
    unknown_features = []
    active_masks = []
    sample_rate = int(artifact["sample_rate"])
    settings = artifact["feature_settings"]
    for record in selected:
        samples, _ = read_wav_mono(esc_root / record.path, sample_rate)
        feature = extract_logmel(samples, sample_rate, settings)
        unknown_features.append(feature)
        total_frames = int(settings["target_frames"])
        active_masks.append(
            frame_rms_dbfs(
                samples,
                sample_rate,
                int(settings["frame_ms"]),
                int(settings["hop_ms"]),
                total_frames,
            )
            >= activity_dbfs
        )
    unknown_logits = _torch_logits(model, np.stack(unknown_features), artifact["normalization"])
    unknown_probabilities = expit(np.clip(unknown_logits / temperature, -40.0, 40.0))
    unknown_scores = unknown_probabilities.max(axis=2)[np.stack(active_masks)]
    sample_count = min(len(known_scores), len(unknown_scores), 100_000)
    known_sample = rng.choice(known_scores, sample_count, replace=False)
    unknown_sample = rng.choice(unknown_scores, sample_count, replace=False)
    grid = np.linspace(0.05, 0.95, 181)
    rows = []
    for threshold in grid:
        known_recall = float(np.mean(known_sample >= threshold))
        unknown_recall = float(np.mean(unknown_sample < threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "known_recall": known_recall,
                "unknown_recall": unknown_recall,
                "balanced_accuracy": (known_recall + unknown_recall) / 2.0,
            }
        )
    best = max(rows, key=lambda row: (row["balanced_accuracy"], row["unknown_recall"]))
    labels = np.concatenate([np.ones(sample_count), np.zeros(sample_count)])
    scores = np.concatenate([known_sample, unknown_sample])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    target_index = int(np.argmin(np.abs(tpr - 0.95)))
    report = _json_ready(
        {
            "method": "maximum_known_probability_frame_rejection",
            "known_source": "DESED validation active labeled frames",
            "unknown_source": "ESC-50 labels outside the deployed 10 classes",
            "unknown_clips": len(selected),
            "balanced_frame_samples_per_side": sample_count,
            "activity_dbfs": activity_dbfs,
            "known_confidence_threshold": best["threshold"],
            "balanced_accuracy": best["balanced_accuracy"],
            "known_recall": best["known_recall"],
            "unknown_recall": best["unknown_recall"],
            "auroc": float(roc_auc_score(labels, scores)),
            "fpr_at_95_tpr": float(fpr[target_index]),
            "threshold_at_95_tpr": float(thresholds[target_index]),
            "threshold_sweep": rows,
            "limitation": (
                "confidence rejection is an open-set baseline, not a learned unknown class"
            ),
        }
    )
    artifact["open_set"] = report
    torch.save(artifact, artifact_path)
    return report


def export_onnx(
    artifact_path: Path,
    output_path: Path,
    verification_feature: np.ndarray,
    opset: int = 17,
) -> dict:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - optional install
        raise RuntimeError("Install onnx and onnxruntime before exporting") from error
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    model = _load_torch_model(artifact)
    frames = int(artifact["feature_settings"]["target_frames"])
    mels = int(artifact["feature_settings"]["n_mels"])
    example = torch.zeros(1, 1, mels, frames, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exporter = "dynamo"
    fallback_reason = None
    try:
        batch = torch.export.Dim("batch")
        program = torch.onnx.export(
            model,
            (example,),
            dynamo=True,
            input_names=["logmel"],
            output_names=["logits"],
            dynamic_shapes=({0: batch},),
            opset_version=opset,
        )
        program.save(output_path)
        exported = onnx.load(str(output_path))
        batch_dimension = exported.graph.input[0].type.tensor_type.shape.dim[0]
        if not batch_dimension.dim_param:
            raise RuntimeError("dynamo exporter specialized the requested dynamic batch")
    except Exception as error:  # pragma: no cover - depends on exporter/RNN support
        exporter = "legacy_fallback"
        fallback_reason = f"{type(error).__name__}: {error}"[:1000]
        torch.onnx.export(
            model,
            example,
            str(output_path),
            dynamo=False,
            input_names=["logmel"],
            output_names=["logits"],
            dynamic_axes={"logmel": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
        )
    checked = onnx.load(str(output_path))
    onnx.checker.check_model(checked)
    graph_input_shape = [
        dimension.dim_param or dimension.dim_value
        for dimension in checked.graph.input[0].type.tensor_type.shape.dim
    ]
    mean, std = (float(value) for value in artifact["normalization"])
    inputs = ((verification_feature - mean) / std)[None, None].astype(np.float32)
    with torch.inference_mode():
        torch_logits = model(torch.from_numpy(inputs)).numpy()
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {session.get_inputs()[0].name: inputs})[0]
    difference = np.abs(torch_logits - onnx_logits)
    batch_two_inputs = np.concatenate([inputs, inputs], axis=0)
    batch_two_output = session.run(None, {session.get_inputs()[0].name: batch_two_inputs})[0]
    report = {
        "artifact": str(artifact_path),
        "onnx": str(output_path),
        "opset": opset,
        "actual_opset": next(
            item.version for item in checked.opset_import if item.domain in ("", "ai.onnx")
        ),
        "exporter": exporter,
        "fallback_reason": fallback_reason,
        "input_shape": list(inputs.shape),
        "graph_input_shape": graph_input_shape,
        "dynamic_batch_verified": bool(
            batch_two_output.shape == (2, frames, len(artifact["class_names"]))
        ),
        "output_shape": list(onnx_logits.shape),
        "maximum_absolute_logit_error": float(difference.max()),
        "mean_absolute_logit_error": float(difference.mean()),
        "allclose_atol_1e-4": bool(np.allclose(torch_logits, onnx_logits, atol=1e-4, rtol=1e-4)),
        "model_bytes": output_path.stat().st_size,
    }
    return report


def quantize_onnx(fp32_path: Path, int8_path: Path, verification_inputs: np.ndarray) -> dict:
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(fp32_path),
        str(int8_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm", "GRU"],
    )
    fp_session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    int_session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    fp_logits = fp_session.run(None, {fp_session.get_inputs()[0].name: verification_inputs})[0]
    int_logits = int_session.run(None, {int_session.get_inputs()[0].name: verification_inputs})[0]
    difference = np.abs(fp_logits - int_logits)
    import onnx

    node_types = [node.op_type for node in onnx.load(str(int8_path)).graph.node]
    quantized_ops = {
        name: node_types.count(name)
        for name in ("DynamicQuantizeLinear", "MatMulInteger", "ConvInteger")
    }
    return {
        "method": "onnxruntime_dynamic_weight_int8_matmul_gemm_gru",
        "fp32_bytes": fp32_path.stat().st_size,
        "int8_bytes": int8_path.stat().st_size,
        "size_reduction_fraction": 1.0 - int8_path.stat().st_size / fp32_path.stat().st_size,
        "quantized_ops": quantized_ops,
        "gru_quantized": False,
        "interpretation": (
            "This runtime converted the output MatMul but retained Conv and GRU in FP32; "
            "size or speed gains are therefore expected to be limited."
        ),
        "maximum_absolute_logit_error": float(difference.max()),
        "mean_absolute_logit_error": float(difference.mean()),
    }


def create_long_demo(
    records: Sequence[SoundscapeRecord],
    dataset_root: Path,
    output_path: Path,
    number_of_clips: int,
) -> tuple[SoundscapeRecord, dict]:
    selected = sorted(records, key=lambda record: record.scene_id)[:number_of_clips]
    if not selected:
        raise ValueError("no records were available for the long-audio demo")
    sample_rate = selected[0].sample_rate
    parts = []
    events = []
    offset_s = 0.0
    sources = []
    for record in selected:
        samples, _ = read_wav_mono(dataset_root / record.path, sample_rate)
        parts.append(samples)
        sources.append(record.scene_id)
        for event in record.events:
            events.append(
                TimedEvent(
                    label=event.label,
                    onset_s=event.onset_s + offset_s,
                    offset_s=event.offset_s + offset_s,
                    source_asset_id=event.source_asset_id,
                    source_recording_id=event.source_recording_id,
                )
            )
        offset_s += len(samples) / sample_rate
    combined = np.concatenate(parts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output_path, sample_rate, combined.astype(np.float32))
    record = SoundscapeRecord(
        scene_id="l4-long-demo",
        path=str(output_path),
        split="engineering",
        duration_s=len(combined) / sample_rate,
        sample_rate=sample_rate,
        events=tuple(events),
    )
    return record, {"sources": sources, "duration_s": record.duration_s, "events": len(events)}


def evaluate_long_audio(
    deployment: SEDDeployment,
    record: SoundscapeRecord,
    audio_path: Path,
    metrics_settings: Mapping[str, Any],
    threshold_profile: str,
    window_seconds: float,
    hop_seconds: float,
) -> tuple[dict, dict]:
    result = deployment.predict_file(
        audio_path,
        threshold_profile=threshold_profile,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
    )
    estimates = [
        TimedEvent(
            label=item["label"],
            onset_s=item["onset_s"],
            offset_s=item["offset_s"],
            source_asset_id="prediction",
            source_recording_id="prediction",
        )
        for item in result["events"]
    ]
    metrics = evaluate_sed(
        [record],
        {record.scene_id: estimates},
        deployment.class_names,
        metrics_settings["event"],
        float(metrics_settings["segment_resolution_s"]),
    )
    public_result = {key: value for key, value in result.items() if key != "probabilities"}
    return metrics, public_result


def evaluate_snr(
    artifact_path: Path,
    records: Sequence[SoundscapeRecord],
    dataset_root: Path,
    snr_values: Sequence[float | None],
    metrics_settings: Mapping[str, Any],
    threshold_profile: str,
    seed: int,
) -> list[dict]:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    model = _load_torch_model(artifact)
    temperature = float(artifact.get("calibration", {}).get("temperature", 1.0))
    threshold = artifact["threshold_profiles"][threshold_profile]
    class_names = artifact["class_names"]
    sample_rate = int(artifact["sample_rate"])
    settings = artifact["feature_settings"]
    rows = []
    for snr_index, snr_db in enumerate(snr_values):
        features = []
        for record_index, record in enumerate(records):
            samples, _ = read_wav_mono(dataset_root / record.path, sample_rate)
            if snr_db is not None:
                samples = add_white_noise(
                    samples, float(snr_db), seed + snr_index * 10_000 + record_index
                )
            features.append(extract_logmel(samples, sample_rate, settings))
        started = time.perf_counter()
        logits = _torch_logits(model, np.stack(features), artifact["normalization"])
        inference_seconds = time.perf_counter() - started
        probabilities = expit(np.clip(logits / temperature, -40.0, 40.0)).astype(np.float32)
        metrics, _ = _evaluate_probabilities(
            records,
            probabilities,
            class_names,
            threshold,
            float(settings["hop_ms"]) / 1000.0,
            artifact["postprocess_settings"],
            metrics_settings,
        )
        rows.append(
            {
                "snr_db": "clean" if snr_db is None else float(snr_db),
                "clips": len(records),
                "event_f1": metrics["event_based"]["micro"]["f1"],
                "macro_event_f1": metrics["event_based"]["macro_f1"],
                "segment_f1": metrics["segment_based"]["micro"]["f1"],
                "model_inference_seconds": inference_seconds,
                "audio_seconds": sum(record.duration_s for record in records),
                "model_only_rtf": inference_seconds
                / max(1e-9, sum(record.duration_s for record in records)),
            }
        )
    return rows


def benchmark_backends(
    artifact_path: Path,
    fp32_onnx: Path,
    int8_onnx: Path,
    feature: np.ndarray,
    long_audio: Path,
    repeats: int,
    window_seconds: float,
    hop_seconds: float,
) -> list[dict]:
    try:
        import psutil

        process = psutil.Process()

        def rss() -> int:
            return int(process.memory_info().rss)

    except ImportError:  # pragma: no cover - optional install

        def rss() -> int:
            return 0

    backends = [
        ("pytorch_fp32", None),
        ("onnxruntime_fp32", fp32_onnx),
        ("onnxruntime_dynamic_int8", int8_onnx),
    ]
    rows = []
    for name, onnx_path in backends:
        before = rss()
        deployment = SEDDeployment(artifact_path, onnx_path)
        after_load = rss()
        for _ in range(3):
            deployment.predict_feature(feature)
        latencies = []
        for _ in range(repeats):
            started = time.perf_counter()
            deployment.predict_feature(feature)
            latencies.append((time.perf_counter() - started) * 1000.0)
        long_result = deployment.predict_file(
            long_audio,
            threshold_profile="overall",
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
            include_unknown=False,
        )
        model_path = artifact_path if onnx_path is None else onnx_path
        if onnx_path is None:
            assert deployment.model is not None
            weight_bytes = sum(
                parameter.numel() * parameter.element_size()
                for parameter in deployment.model.parameters()
            )
        else:
            import onnx
            from onnx import numpy_helper

            graph = onnx.load(str(onnx_path)).graph
            weight_bytes = sum(
                numpy_helper.to_array(initializer).nbytes for initializer in graph.initializer
            )
        rows.append(
            {
                "backend": name,
                "model_bytes": model_path.stat().st_size,
                "weight_bytes": weight_bytes,
                "rss_load_delta_bytes_best_effort": max(0, after_load - before),
                "window_latency_ms_median": float(np.median(latencies)),
                "window_latency_ms_p95": float(np.percentile(latencies, 95)),
                "model_window_rtf_median": float(np.median(latencies)) / (window_seconds * 1000.0),
                "long_audio_seconds": long_result["stats"]["audio_seconds"],
                "long_end_to_end_seconds": long_result["stats"]["total_seconds"],
                "long_end_to_end_rtf": long_result["stats"]["rtf"],
            }
        )
        del deployment
    return rows


def run_l4_engineering(config: Mapping[str, Any], root: Path) -> dict:
    paths = config["paths"]
    artifact_root = root / paths["artifact_dir"]
    report_root = root / paths["report_dir"]
    demo_root = root / paths["demo_dir"]
    artifact_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    demo_root.mkdir(parents=True, exist_ok=True)
    deployment_artifact = artifact_root / "crnn_short_event_calibrated.pt"
    calibration = calibrate_artifact(
        root / paths["source_artifact"],
        root / paths["validation_cache"],
        root / paths["manifest"],
        deployment_artifact,
        config["calibration"]["threshold_grid"],
        config["metrics"],
    )
    open_set = calibrate_open_set(
        deployment_artifact,
        root / paths["validation_cache"],
        root / paths["esc_manifest"],
        root / paths["esc_root"],
        int(config["open_set"]["maximum_unknown_clips"]),
        float(config["open_set"]["activity_dbfs"]),
        int(config["project"]["seed"]),
    )
    validation_feature = np.load(root / paths["validation_cache"])["features"][0]
    fp32_onnx = artifact_root / "crnn_short_event_fp32.onnx"
    onnx_report = export_onnx(
        deployment_artifact,
        fp32_onnx,
        validation_feature,
        int(config["export"]["opset"]),
    )
    artifact = torch.load(deployment_artifact, map_location="cpu", weights_only=True)
    mean, std = (float(value) for value in artifact["normalization"])
    verification_inputs = ((validation_feature - mean) / std)[None, None].astype(np.float32)
    int8_onnx = artifact_root / "crnn_short_event_dynamic_int8.onnx"
    quantization = quantize_onnx(fp32_onnx, int8_onnx, verification_inputs)

    all_records = read_soundscape_manifest(root / paths["manifest"])
    test_records = [record for record in all_records if record.split == "test"]
    long_record, long_metadata = create_long_demo(
        test_records,
        root / paths["dataset_root"],
        demo_root / "l4_long_demo.wav",
        int(config["long_audio"]["clips"]),
    )
    deployment = SEDDeployment(deployment_artifact)
    long_metrics, long_prediction = evaluate_long_audio(
        deployment,
        long_record,
        demo_root / "l4_long_demo.wav",
        config["metrics"],
        str(config["deployment"]["threshold_profile"]),
        float(config["long_audio"]["window_seconds"]),
        float(config["long_audio"]["hop_seconds"]),
    )
    stream_result = stream_file(
        demo_root / "l4_long_demo.wav",
        deployment_artifact,
        threshold_profile=str(config["deployment"]["threshold_profile"]),
        chunk_seconds=float(config["streaming"]["chunk_seconds"]),
        onnx_path=fp32_onnx,
    )
    snr_rows = evaluate_snr(
        deployment_artifact,
        test_records,
        root / paths["dataset_root"],
        [None if value == "clean" else float(value) for value in config["robustness"]["snr_db"]],
        config["metrics"],
        str(config["deployment"]["threshold_profile"]),
        int(config["project"]["seed"]),
    )
    benchmark = benchmark_backends(
        deployment_artifact,
        fp32_onnx,
        int8_onnx,
        validation_feature,
        demo_root / "l4_long_demo.wav",
        int(config["benchmark"]["repeats"]),
        float(config["long_audio"]["window_seconds"]),
        float(config["long_audio"]["hop_seconds"]),
    )
    summary = {
        "protocol": "l4-long-stream-open-set-onnx-cpu-engineering-v1",
        "calibration": calibration,
        "open_set": open_set,
        "onnx": onnx_report,
        "quantization": quantization,
        "long_audio": {
            "metadata": long_metadata,
            "metrics": long_metrics,
            "prediction": long_prediction,
        },
        "streaming": stream_result,
        "snr_robustness": snr_rows,
        "benchmark": benchmark,
    }
    _write_json(report_root / "l4_summary.json", summary)
    _write_json(report_root / "calibration.json", calibration)
    _write_json(report_root / "open_set.json", open_set)
    _write_json(report_root / "onnx_verification.json", onnx_report)
    _write_json(report_root / "quantization.json", quantization)
    _write_json(report_root / "long_audio_prediction.json", long_prediction)
    _write_json(report_root / "streaming_prediction.json", stream_result)
    with (report_root / "snr_robustness.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(snr_rows[0]))
        writer.writeheader()
        writer.writerows(snr_rows)
    with (report_root / "benchmark.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(benchmark[0]))
        writer.writeheader()
        writer.writerows(benchmark)
    shutil.copy2(config["config_path"], report_root / "resolved_config.yaml")
    return summary
