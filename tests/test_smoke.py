from audioscope.cli import run_cross_validation, run_smoke


def test_end_to_end_smoke(tmp_path):
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    config = project / "configs" / "smoke.yaml"
    config.write_text(
        """
project: {seed: 2027}
data:
  root: data/smoke
  manifest: data/smoke/manifest.jsonl
  sample_rate: 8000
  duration_s: 0.15
  test_fold: 5
features: {n_mfcc: 12, n_mels: 24, n_fft: 256, frame_ms: 25, hop_ms: 10}
model: {c: 10.0, kernel: rbf}
outputs: {artifact_dir: artifacts/smoke, report_dir: reports/generated/smoke}
robustness: {snr_db: [10, 0]}
""".strip(),
        encoding="utf-8",
    )
    result = run_smoke(config)
    assert result["validation"]["records"] == 30
    assert result["metrics"]["macro_f1"] >= 0.9
    assert (project / "artifacts" / "smoke" / "model.joblib").is_file()
    assert (project / "reports" / "generated" / "smoke" / "confusion_matrix.csv").is_file()

    crossval = run_cross_validation(config)
    assert len(crossval["folds"]) == 5
    assert crossval["aggregate"]["macro_f1"]["mean"] >= 0.9
