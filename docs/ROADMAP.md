# AudioScope roadmap

## Post-v1.0 workbench enhancement

- [x] Three-seed paired augmentation experiment and explicit negative-result retention
- [x] Validation-only decoder selection, frozen 221-clip/seven-condition test, duration/error slices
- [x] Candidate 1,800-second bounded streaming regression and explicit default-model registry
- [x] Replay, append-only corrections, aligned reference timelines and same-audio version comparisons
- [x] Local history, privacy defaults, relinking, JSON/CSV exports and friendly validation messages
- [x] Known-label overlap queries and locally evaluated English CLAP retrieval
- [x] Formal CPU Python container, fresh/repeated PostgreSQL migration and real workflow acceptance
- [x] 75 automated regression tests; source-hash and dependency snapshot verification
- Final enhancement audit/catalog/archive: independently verified by the three reports under
  `reports/generated/workbench-release/`; homepage and usage documentation updated.

Usage: `docs/WORKBENCH_GUIDE.md`; evidence and limitations: `reports/WORKBENCH_IMPROVEMENT.md`.
Physical certification remains unperformed. Review does not automatically enter training data.

## v1.0 software release (2026-10-11)

- [x] Seven deterministic software shifts on 221 test recordings, repeated inference verification
- [x] Bounded incremental event decoder and 10,000-block state regression test
- [x] Accelerated 1,800-second audio streaming acceptance
- [x] Generated Chinese-narrated video with Chinese/English subtitles
- [x] Separate software-final and field-final release policies

The unchecked physical capture items below are optional post-v1.0 work. They do not block software-final.

## Week 1 - infrastructure and honest baseline

- [x] Repository structure, dependency metadata and CI
- [x] Dataset source/license registry
- [x] JSONL audio manifest and checksum validation
- [x] Source-recording leakage guard
- [x] Deterministic synthetic end-to-end smoke dataset
- [x] EDA for class/fold balance, duration, level and anomalies
- [x] MFCC summary + SVM training/evaluation path
- [x] Register and prepare a local ESC-50 copy
- [x] Run all five official folds and freeze baseline metrics
- [x] Record the four source IDs that overlap official folds as a comparability caveat

## Week 2 - classification MVP

- [x] log-mel CNN with fixed seeds and augmentation controls
- [x] pretrained AST frozen-encoder linear probe and sample-efficiency curve
- [x] CLAP fixed-prompt zero-shot evaluation and frozen-encoder linear probe
- [x] white-noise robustness curves, including a waveform-noise training ablation
- [x] deterministic room/device capture kit, synchronization, QC and metric pipeline
- [ ] execute the physical room/device recordings (现场操作)
- [x] per-class errors and confusion matrices
- [x] upload/recording demo backed by the real checkpoint

## Weeks 3-4 - complete AudioScope

- [x] source-disjoint frame-labelled polyphonic ESC-10 soundscape dataset
- [x] CRNN baseline with event-F1, segment-F1 and validation-only threshold tuning
- [x] real strongly annotated DESED validation with source-disjoint splits
- [x] global, class-balanced and short-event validation-only threshold profiles
- [x] short-event weighted loss and duration-stratified evaluation
- [x] CRNN versus lightweight Conformer accuracy/cost comparison
- [x] confidence calibration and open-set evaluation
- [x] long-audio timeline and streaming window merge
- [x] ONNX export plus CPU latency, RTF and memory report
- [x] deployment model card and interactive upload/recording demo
- [x] PostgreSQL lineage schema, Alembic migration and idempotent import
- [x] Docker Compose database/API/demo deployment
- [x] bilingual project summary, resume bullets and demo-video script
- [ ] record the physical microphone acceptance session (现场操作)
- [x] generate software-validation demo video; optional manual field video remains future work
