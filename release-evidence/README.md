# Public release evidence

These machine-readable files are copied from the verified local release outputs. They contain no raw audio,
model weights, credentials, database contents, or user review history.

- `robustness.json`: all 221 DESED test clips across seven deterministic software conditions;
- `stream_soak.json`: bounded 1,800-second accelerated file-stream test;
- `video_validation.json`: technical checks for the generated demonstration video;
- `workbench_final_audit.json`: final workbench enhancement audit;
- `model_registry.json`: relative model paths and hashes used by the local workbench.

The evidence explicitly records `physical_validation_performed: false`. Software-domain-shift results must not
be described as physical-device or real-room certification.
