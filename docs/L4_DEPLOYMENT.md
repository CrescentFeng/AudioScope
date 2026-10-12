# AudioScope L4 deployment guide

## Artifacts

- `crnn_short_event_calibrated.pt`: metadata source of truth, including classes, normalization,
  temperature, threshold profiles, post-processing and open-set threshold.
- `crnn_short_event_fp32.onnx`: recommended CPU graph, opset 17, dynamic batch.
- `crnn_short_event_dynamic_int8.onnx`: reproducible negative quantization experiment; not the
  recommended runtime artifact.

The ONNX graph accepts normalized log-mel tensors shaped `[batch, 1, 64, 500]` and returns logits
shaped `[batch, 500, 10]`. Waveform resampling, log-mel extraction, temperature scaling and event
post-processing remain in Python so every backend uses the same contract.

## Install and rebuild

```bash
.venv/bin/python -m pip install -e '.[engineering,demo,microphone,dev]'
.venv/bin/audioscope l4-build --config configs/l4_engineering.yaml
```

The build is deterministic with respect to data selection and noise seeds. Runtime timings vary by
CPU load and power state.

## Long files

```bash
.venv/bin/audioscope sed-long-predict \
  --model artifacts/l4-engineering/crnn_short_event_calibrated.pt \
  --onnx artifacts/l4-engineering/crnn_short_event_fp32.onnx \
  --audio input.wav --threshold-profile overall
```

Use `short_event` only when higher short-event sensitivity is the intended operating point.
`class_balanced` is retained for analysis and is not the default production profile.

## Stream simulation and microphone

```bash
.venv/bin/audioscope sed-stream-file \
  --model artifacts/l4-engineering/crnn_short_event_calibrated.pt \
  --onnx artifacts/l4-engineering/crnn_short_event_fp32.onnx \
  --audio input.wav --chunk-seconds 1

.venv/bin/audioscope sed-stream-microphone \
  --model artifacts/l4-engineering/crnn_short_event_calibrated.pt \
  --onnx artifacts/l4-engineering/crnn_short_event_fp32.onnx \
  --duration-seconds 30 --chunk-seconds 1
```

The microphone command needs OS microphone permission and a working PortAudio device. A 1-second
chunk means approximately 1 second of acquisition latency plus a few milliseconds of compute. The
processor retains only 10 seconds of waveform history.

## Demo

```bash
.venv/bin/python l4_app.py \
  --model artifacts/l4-engineering/crnn_short_event_calibrated.pt \
  --onnx artifacts/l4-engineering/crnn_short_event_fp32.onnx
```

Open `http://127.0.0.1:7860`. The UI supports uploads and browser microphone recordings, threshold
profiles, long-file inference, stream simulation, an event table, timeline and engineering stats.

## Operational checks

Before distributing a model build:

1. run `ruff check .`, `pytest` and `pip check`;
2. confirm ONNX checker, batch=2 execution and numerical consistency in `onnx_verification.json`;
3. inspect clean and SNR metrics rather than latency alone;
4. keep the `.pt` metadata file next to ONNX, because ONNX does not contain the full event-decoding
   contract;
5. treat `unknown` as a rejection signal requiring review, not as a semantic class.
