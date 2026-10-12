from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Optional

from .config import load_config, resolve
from .eda import build_eda
from .esc50 import prepare_esc50
from .experiment import evaluate, load_artifact, predict_file, train
from .manifest import read_manifest, validate_manifest
from .synthetic import generate_smoke_dataset


def _paths(config_path: Path) -> tuple[Dict[str, Any], Path, Path, Path]:
    config_path = config_path.resolve()
    root = config_path.parent.parent
    config = load_config(config_path)
    data_root = resolve(root, config["data"]["root"])
    manifest = resolve(root, config["data"]["manifest"])
    return config, root, data_root, manifest


def _cache_dir(config: Dict[str, Any], root: Path) -> Optional[Path]:
    value = config["data"].get("feature_cache")
    return resolve(root, value) if value else None


def _validate(config: Dict[str, Any], records, data_root: Path) -> dict:
    return validate_manifest(
        records,
        data_root,
        allow_source_fold_overlap=bool(config["data"].get("allow_source_fold_overlap", False)),
    )


def run_smoke(config_path: Path) -> dict:
    config, root, data_root, manifest = _paths(config_path)
    generate_smoke_dataset(
        data_root,
        int(config["data"]["sample_rate"]),
        float(config["data"]["duration_s"]),
        int(config["project"]["seed"]),
    )
    records = read_manifest(manifest)
    validation = _validate(config, records, data_root)
    report_dir = resolve(root, config["outputs"]["report_dir"])
    eda = build_eda(records, data_root, report_dir)
    artifact_dir = resolve(root, config["outputs"]["artifact_dir"])
    artifact_path = artifact_dir / "model.joblib"
    cache_dir = _cache_dir(config, root)
    training = train(
        records,
        data_root,
        int(config["data"]["sample_rate"]),
        config["features"],
        config["model"],
        int(config["data"]["test_fold"]),
        int(config["project"]["seed"]),
        artifact_path,
        cache_dir,
    )
    metrics = evaluate(
        records,
        data_root,
        load_artifact(artifact_path),
        report_dir,
        config["robustness"]["snr_db"],
        int(config["project"]["seed"]),
        cache_dir,
    )
    result = {"validation": validation, "eda": eda, "training": training, "metrics": metrics}
    (report_dir / "smoke_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def run_cross_validation(config_path: Path) -> dict:
    config, root, data_root, manifest = _paths(config_path)
    records = read_manifest(manifest)
    validation = _validate(config, records, data_root)
    excluded_source_ids = []
    excluded_records = 0
    if config["data"].get("exclude_source_fold_overlaps", False):
        excluded_source_ids = list(validation["source_fold_overlaps"])
        excluded_set = set(excluded_source_ids)
        original_count = len(records)
        records = [record for record in records if record.source_recording_id not in excluded_set]
        excluded_records = original_count - len(records)
    folds = [int(value) for value in config["data"].get("folds", sorted({r.fold for r in records}))]
    available_folds = {record.fold for record in records}
    if not folds or not set(folds).issubset(available_folds):
        raise ValueError(
            f"Requested folds {folds} do not match available folds {sorted(available_folds)}"
        )
    artifact_root = resolve(root, config["outputs"]["artifact_dir"])
    report_root = resolve(root, config["outputs"]["report_dir"])
    cache_dir = _cache_dir(config, root)
    fold_results = []
    for fold in folds:
        artifact_path = artifact_root / f"fold-{fold}" / "model.joblib"
        report_dir = report_root / f"fold-{fold}"
        training = train(
            records,
            data_root,
            int(config["data"]["sample_rate"]),
            config["features"],
            config["model"],
            fold,
            int(config["project"]["seed"]),
            artifact_path,
            cache_dir,
        )
        metrics = evaluate(
            records,
            data_root,
            load_artifact(artifact_path),
            report_dir,
            config["robustness"]["snr_db"],
            int(config["project"]["seed"]),
            cache_dir,
        )
        fold_results.append({"fold": fold, "training": training, "metrics": metrics})
    aggregate = {}
    for metric in ("accuracy", "macro_f1"):
        values = [item["metrics"][metric] for item in fold_results]
        aggregate[metric] = {"mean": mean(values), "std": pstdev(values), "values": values}
    aggregate["robustness"] = []
    for snr_db in config["robustness"]["snr_db"]:
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
        "protocol": (
            "official-fold-cross-validation-with-overlap-sources-excluded"
            if excluded_source_ids
            else "official-fold-cross-validation"
        ),
        "validation": validation,
        "evaluated_records": len(records),
        "excluded_source_ids": excluded_source_ids,
        "excluded_records": excluded_records,
        "folds": fold_results,
        "aggregate": aggregate,
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "cross_validation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def run_cnn(config_path: Path) -> dict:
    from .cnn import run_cnn_cross_validation

    config, root, data_root, manifest = _paths(config_path)
    if config["data"].get("generated", False):
        generate_smoke_dataset(
            data_root,
            int(config["data"]["sample_rate"]),
            float(config["data"]["duration_s"]),
            int(config["project"]["seed"]),
        )
    records = read_manifest(manifest)
    _validate(config, records, data_root)
    return run_cnn_cross_validation(
        records=records,
        data_root=data_root,
        sample_rate=int(config["data"]["sample_rate"]),
        feature_settings=config["features"],
        model_settings=config["model"],
        folds=config["data"].get("folds", sorted({record.fold for record in records})),
        snr_values=config["robustness"]["snr_db"],
        seed=int(config["project"]["seed"]),
        artifact_root=resolve(root, config["outputs"]["artifact_dir"]),
        report_root=resolve(root, config["outputs"]["report_dir"]),
        cache_dir=_cache_dir(config, root),
    )


def run_transfer(config_path: Path) -> dict:
    from .transfer import run_ast_transfer, run_clap_transfer

    config, root, data_root, manifest = _paths(config_path)
    records = read_manifest(manifest)
    _validate(config, records, data_root)
    family = str(config["transfer"]["family"])
    transfer_settings = dict(config["transfer"])
    if transfer_settings.get("model_path"):
        transfer_settings["model_path"] = str(resolve(root, transfer_settings["model_path"]))
    common = {
        "records": records,
        "data_root": data_root,
        "folds": config["data"].get("folds", sorted({record.fold for record in records})),
        "transfer_settings": transfer_settings,
        "probe_settings": config["probe"],
        "seed": int(config["project"]["seed"]),
        "cache_dir": resolve(root, config["outputs"]["embedding_cache"]),
        "artifact_root": resolve(root, config["outputs"]["artifact_dir"]),
        "report_root": resolve(root, config["outputs"]["report_dir"]),
    }
    if family == "ast":
        return run_ast_transfer(**common)
    if family == "clap":
        return run_clap_transfer(zero_shot_settings=config["zero_shot"], **common)
    raise ValueError(f"Unsupported transfer family: {family}")


def run_sed(config_path: Path) -> dict:
    from .sed import run_sed_experiment

    config_path = config_path.resolve()
    root = config_path.parent.parent
    config = load_config(config_path)
    source_root = resolve(root, config["source_data"]["root"])
    source_manifest = resolve(root, config["source_data"]["manifest"])
    source_records = read_manifest(source_manifest)
    return run_sed_experiment(
        source_records=source_records,
        source_root=source_root,
        soundscape_root=resolve(root, config["soundscapes"]["root"]),
        generation_settings=config["soundscapes"],
        feature_settings=config["features"],
        model_settings=config["model"],
        postprocess_settings=config["postprocess"],
        metrics_settings=config["metrics"],
        seed=int(config["project"]["seed"]),
        artifact_root=resolve(root, config["outputs"]["artifact_dir"]),
        report_root=resolve(root, config["outputs"]["report_dir"]),
        cache_root=resolve(root, config["outputs"]["feature_cache"]),
    )


def run_sed_enhancement(config_path: Path) -> dict:
    from .sed_enhanced import run_desed_enhancement

    config_path = config_path.resolve()
    root = config_path.parent.parent
    config = load_config(config_path)
    return run_desed_enhancement(
        dataset_settings=config["dataset"],
        feature_settings=config["features"],
        base_model_settings=config["model"],
        variants=config["comparison"]["variants"],
        postprocess_settings=config["postprocess"],
        metrics_settings=config["metrics"],
        seed=int(config["project"]["seed"]),
        shard_root=resolve(root, config["dataset"]["shard_root"]),
        dataset_root=resolve(root, config["dataset"]["prepared_root"]),
        cache_root=resolve(root, config["outputs"]["feature_cache"]),
        artifact_root=resolve(root, config["outputs"]["artifact_dir"]),
        report_root=resolve(root, config["outputs"]["report_dir"]),
    )


def run_l4(config_path: Path) -> dict:
    from .engineering_eval import run_l4_engineering

    config_path = config_path.resolve()
    root = config_path.parent.parent
    config = load_config(config_path)
    config["config_path"] = str(config_path)
    return run_l4_engineering(config, root)


def main() -> None:
    parser = argparse.ArgumentParser(prog="audioscope")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser(
        "smoke", help="Run the deterministic end-to-end smoke experiment"
    )
    smoke_parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    validate_parser = subparsers.add_parser("validate-manifest")
    validate_parser.add_argument("--config", type=Path, required=True)
    eda_parser = subparsers.add_parser("eda")
    eda_parser.add_argument("--config", type=Path, required=True)
    esc_parser = subparsers.add_parser("prepare-esc50")
    esc_parser.add_argument("--dataset-root", type=Path, required=True)
    esc_parser.add_argument("--output", type=Path, default=Path("data/manifests/esc50.jsonl"))
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)
    crossval_parser = subparsers.add_parser("cross-validate")
    crossval_parser.add_argument("--config", type=Path, required=True)
    cnn_parser = subparsers.add_parser("cnn-cross-validate")
    cnn_parser.add_argument("--config", type=Path, required=True)
    transfer_parser = subparsers.add_parser("transfer-evaluate")
    transfer_parser.add_argument("--config", type=Path, required=True)
    sed_parser = subparsers.add_parser("sed-train-evaluate")
    sed_parser.add_argument("--config", type=Path, required=True)
    sed_enhance_parser = subparsers.add_parser("sed-enhance-evaluate")
    sed_enhance_parser.add_argument("--config", type=Path, required=True)
    l4_parser = subparsers.add_parser("l4-build", help="Build and verify the L4 deployment package")
    l4_parser.add_argument("--config", type=Path, default=Path("configs/l4_engineering.yaml"))
    sed_predict_parser = subparsers.add_parser("sed-predict")
    sed_predict_parser.add_argument("--model", type=Path, required=True)
    sed_predict_parser.add_argument("--audio", type=Path, required=True)
    sed_predict_parser.add_argument(
        "--threshold-profile",
        choices=("overall", "class_balanced", "short_event"),
        default="overall",
    )
    long_parser = subparsers.add_parser("sed-long-predict")
    long_parser.add_argument("--model", type=Path, required=True)
    long_parser.add_argument("--audio", type=Path, required=True)
    long_parser.add_argument("--onnx", type=Path)
    long_parser.add_argument("--window-seconds", type=float, default=10.0)
    long_parser.add_argument("--hop-seconds", type=float, default=5.0)
    long_parser.add_argument(
        "--threshold-profile",
        choices=("overall", "class_balanced", "short_event"),
        default="overall",
    )
    stream_file_parser = subparsers.add_parser("sed-stream-file")
    stream_file_parser.add_argument("--model", type=Path, required=True)
    stream_file_parser.add_argument("--audio", type=Path, required=True)
    stream_file_parser.add_argument("--onnx", type=Path)
    stream_file_parser.add_argument("--chunk-seconds", type=float, default=1.0)
    stream_file_parser.add_argument(
        "--threshold-profile",
        choices=("overall", "class_balanced", "short_event"),
        default="overall",
    )
    microphone_parser = subparsers.add_parser("sed-stream-microphone")
    microphone_parser.add_argument("--model", type=Path, required=True)
    microphone_parser.add_argument("--onnx", type=Path)
    microphone_parser.add_argument("--duration-seconds", type=float, default=10.0)
    microphone_parser.add_argument("--chunk-seconds", type=float, default=1.0)
    microphone_parser.add_argument(
        "--threshold-profile",
        choices=("overall", "class_balanced", "short_event"),
        default="overall",
    )
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--config", type=Path, required=True)
    eval_parser.add_argument("--model", type=Path, required=True)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--audio", type=Path, required=True)
    cnn_predict_parser = subparsers.add_parser("cnn-predict")
    cnn_predict_parser.add_argument("--model", type=Path, required=True)
    cnn_predict_parser.add_argument("--audio", type=Path, required=True)
    clean_parser = subparsers.add_parser("clean-generated")
    clean_parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "smoke":
        print(json.dumps(run_smoke(args.config), indent=2, ensure_ascii=False))
        return
    if args.command == "prepare-esc50":
        print(prepare_esc50(args.dataset_root.resolve(), args.output.resolve()))
        return
    if args.command == "cross-validate":
        print(json.dumps(run_cross_validation(args.config), indent=2, ensure_ascii=False))
        return
    if args.command == "cnn-cross-validate":
        print(json.dumps(run_cnn(args.config), indent=2, ensure_ascii=False))
        return
    if args.command == "transfer-evaluate":
        print(json.dumps(run_transfer(args.config), indent=2, ensure_ascii=False))
        return
    if args.command == "sed-train-evaluate":
        print(json.dumps(run_sed(args.config), indent=2, ensure_ascii=False))
        return
    if args.command == "sed-enhance-evaluate":
        print(json.dumps(run_sed_enhancement(args.config), indent=2, ensure_ascii=False))
        return
    if args.command == "l4-build":
        print(json.dumps(run_l4(args.config), indent=2, ensure_ascii=False))
        return
    if args.command == "sed-predict":
        from .sed import predict_sed_file

        print(
            json.dumps(
                predict_sed_file(
                    args.audio.resolve(),
                    args.model.resolve(),
                    args.threshold_profile,
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.command == "sed-long-predict":
        from .engineering import SEDDeployment

        result = SEDDeployment(
            args.model.resolve(), args.onnx.resolve() if args.onnx else None
        ).predict_file(
            args.audio.resolve(),
            threshold_profile=args.threshold_profile,
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
        )
        result.pop("probabilities", None)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if args.command == "sed-stream-file":
        from .engineering import stream_file

        print(
            json.dumps(
                stream_file(
                    args.audio.resolve(),
                    args.model.resolve(),
                    threshold_profile=args.threshold_profile,
                    chunk_seconds=args.chunk_seconds,
                    onnx_path=args.onnx.resolve() if args.onnx else None,
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.command == "sed-stream-microphone":
        from .engineering import stream_microphone

        print(
            json.dumps(
                stream_microphone(
                    args.model.resolve(),
                    duration_seconds=args.duration_seconds,
                    threshold_profile=args.threshold_profile,
                    chunk_seconds=args.chunk_seconds,
                    onnx_path=args.onnx.resolve() if args.onnx else None,
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.command == "predict":
        print(
            json.dumps(
                predict_file(args.audio.resolve(), load_artifact(args.model.resolve()))[:3],
                indent=2,
            )
        )
        return
    if args.command == "cnn-predict":
        from .cnn import predict_cnn_file

        print(
            json.dumps(
                predict_cnn_file(args.audio.resolve(), args.model.resolve())[:3],
                indent=2,
            )
        )
        return
    config, root, data_root, manifest = _paths(args.config)
    if args.command == "clean-generated":
        if not config["data"].get("generated", False):
            raise ValueError("Refusing to delete a data root that is not marked generated: true")
        for path in (
            data_root,
            resolve(root, config["outputs"]["artifact_dir"]),
            resolve(root, config["outputs"]["report_dir"]),
        ):
            if path.exists():
                shutil.rmtree(path)
        print("Generated smoke outputs removed.")
        return
    records = read_manifest(manifest)
    if args.command == "validate-manifest":
        result = _validate(config, records, data_root)
    elif args.command == "eda":
        result = build_eda(records, data_root, resolve(root, config["outputs"]["report_dir"]))
    elif args.command == "train":
        artifact_path = resolve(root, config["outputs"]["artifact_dir"]) / "model.joblib"
        result = train(
            records,
            data_root,
            int(config["data"]["sample_rate"]),
            config["features"],
            config["model"],
            int(config["data"]["test_fold"]),
            int(config["project"]["seed"]),
            artifact_path,
            _cache_dir(config, root),
        )
    elif args.command == "evaluate":
        result = evaluate(
            records,
            data_root,
            load_artifact(args.model.resolve()),
            resolve(root, config["outputs"]["report_dir"]),
            config["robustness"]["snr_db"],
            int(config["project"]["seed"]),
            _cache_dir(config, root),
        )
    else:
        raise RuntimeError(f"Unhandled command: {args.command}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
