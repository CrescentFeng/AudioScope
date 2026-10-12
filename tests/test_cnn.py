import pytest

pytest.importorskip("torch")

from audioscope.cli import run_cnn
from audioscope.cnn import predict_cnn_file


def test_cnn_smoke_cross_validation(tmp_path):
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    config = project / "configs" / "cnn_smoke.yaml"
    config.write_text(
        """
project: {seed: 2027}
data:
  generated: true
  root: data/smoke
  manifest: data/smoke/manifest.jsonl
  sample_rate: 8000
  duration_s: 0.25
  folds: [1, 2]
  feature_cache: data/processed/cnn-smoke-cache
features: {n_mels: 24, n_fft: 256, frame_ms: 25, hop_ms: 10, target_frames: 24}
model:
  device: cpu
  channels: [4, 8]
  dropout: 0.1
  batch_size: 8
  epochs: 12
  learning_rate: 0.003
  weight_decay: 0.0001
  label_smoothing: 0.0
  augmentation: {max_time_shift_frames: 2, frequency_mask: 2, time_mask: 2}
outputs: {artifact_dir: artifacts/cnn-smoke, report_dir: reports/cnn-smoke}
robustness: {snr_db: [10]}
""".strip(),
        encoding="utf-8",
    )
    result = run_cnn(config)
    assert result["aggregate"]["macro_f1"]["mean"] >= 0.9
    model_path = project / "artifacts" / "cnn-smoke" / "fold-1" / "model.pt"
    assert model_path.is_file()
    audio_path = project / "data" / "smoke" / "low_tone" / "low_tone-00.wav"
    prediction = predict_cnn_file(audio_path, model_path)
    assert prediction[0]["label"] == "low_tone"
    assert sum(item["probability"] for item in prediction) == pytest.approx(1.0)
