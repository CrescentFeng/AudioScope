# Data policy

Audio files are intentionally excluded from Git. Every dataset must be registered in
`sources.yaml` before download and represented by a JSONL manifest before training.

Required manifest fields:

- `asset_id`, `path`, `sha256`
- `dataset`, `dataset_version`, `source_url`, `license`
- `sample_rate`, `channels`, `duration_s`
- `label`, `fold`, `source_recording_id`

`path` is relative to the configured dataset root. Derived clips must retain the same
`source_recording_id` as their parent, so leakage checks can reject cross-fold copies.

ESC-50 is licensed CC BY-NC 3.0. Do not imply commercial training rights. The locally
generated smoke set is only a software test fixture and must never be reported as a model result.

