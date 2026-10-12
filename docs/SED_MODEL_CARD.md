# AudioScope L3 CRNN model card

## Model

A 246,762-parameter CRNN consumes 64-bin log-mel features at a 20 ms hop. Three convolutional blocks
downsample frequency only; a bidirectional GRU preserves the 500-frame time axis and predicts ten
independent sigmoid outputs per frame.

Training uses frame-level binary cross entropy, class-positive weighting capped at 10, AdamW,
frequency/time masking and validation-loss early stopping. The best checkpoint is epoch 22 of 29.

## Post-processing and metrics

- Global threshold 0.8, selected only on validation data.
- Five-frame median smoothing, 0.6-second gap merge and 0.4-second minimum event duration.
- Event matching: 0.2-second onset collar; offset tolerance is max(0.2 seconds, 20% of reference
  duration).
- Segment metric: one-second activity grid.

| Test metric | Score |
|---|---:|
| Micro event-F1 | 0.248 |
| Macro event-F1 | 0.241 |
| Micro segment-F1 | 0.557 |
| Macro segment-F1 | 0.539 |

## Intended use and limitations

The model is a reproducible first baseline for studying overlapping event localization and
post-processing. It is not calibrated, open-set, streaming, or validated on real strongly annotated
soundscapes. `sneezing`, `clock_tick` and `rooster` have very weak event-level results, so the model
must not be used for safety, monitoring or production decisions.
