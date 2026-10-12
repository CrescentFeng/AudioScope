# ESC-50 LogMel CNN baseline

Experiment date: 2026-09-15  
Dataset revision: `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`  
Dataset license: CC BY-NC 3.0  
Runtime: Python 3.12.14, PyTorch 2.14.0, CPU

## Protocol

- 2,000 five-second clips, 50 balanced classes, official five folds.
- Audio is converted to mono and resampled from 44.1 kHz to 32 kHz.
- Input is a fixed `64 × 250` log-mel image using a 25 ms frame and 20 ms hop.
- The network uses three Conv2d-BatchNorm-ReLU-MaxPool blocks with 16, 32 and 64 channels,
  frequency-preserving adaptive pooling, dropout and a 50-class linear head.
- The model has 36,370 trainable parameters.
- Every fold uses 15 fixed epochs, AdamW, label smoothing, time shift, frequency masking and time
  masking. The test fold is never used for fitting or model selection.
- A second controlled run adds 20 and 10 dB waveform-noise variants to 50% of training samples.

## Clean CNN results

| Test fold | Accuracy | Macro-F1 |
|---:|---:|---:|
| 1 | 0.483 | 0.454 |
| 2 | 0.480 | 0.459 |
| 3 | 0.455 | 0.430 |
| 4 | 0.548 | 0.516 |
| 5 | 0.560 | 0.541 |
| **Mean ± population SD** | **0.505 ± 0.041** | **0.480 ± 0.041** |

Compared with MFCC + SVM, the clean CNN improves mean Accuracy by 13.2 percentage points and
Macro-F1 by 16.8 points. Five CPU training runs took 288.9 seconds in total after feature extraction.

## Noise-augmented CNN results

| Test fold | Accuracy | Macro-F1 |
|---:|---:|---:|
| 1 | 0.448 | 0.430 |
| 2 | 0.493 | 0.473 |
| 3 | 0.465 | 0.435 |
| 4 | 0.518 | 0.483 |
| 5 | 0.540 | 0.521 |
| **Mean ± population SD** | **0.493 ± 0.034** | **0.468 ± 0.033** |

Noise augmentation costs 1.2 Macro-F1 points on clean audio while materially improving robustness:

| Model | Clean | 20 dB | 10 dB | 0 dB |
|---|---:|---:|---:|---:|
| MFCC + SVM | 0.312 | 0.225 | 0.130 | 0.032 |
| Clean LogMel CNN | **0.480** | 0.203 | 0.057 | 0.020 |
| Noise-augmented LogMel CNN | 0.468 | **0.416** | **0.323** | **0.101** |

The clean CNN is the headline clean-condition baseline. The noise-augmented CNN is the default
robustness candidate because its 20 and 10 dB results improve substantially with a small clean-score
tradeoff. Both runs remain part of the experiment record.

## Error analysis

For the clean CNN, frequent directional confusions include `coughing → sneezing` (10),
`breathing → crow` (9), `siren → church_bells` (9), `train → sea_waves` (9) and
`can_opening → glass_breaking` (8). `door_wood_creaks` remains the weakest class, with 1/40 correct.

The noise-augmented model changes the error distribution rather than uniformly solving it. Its common
confusions include `siren → church_bells` (14), `cow → church_bells` (11), `rain → sea_waves` (11)
and `can_opening → glass_breaking` (10). These pairs should drive the next augmentation and pretrained
embedding experiments.

## Limitations

- Robustness currently uses white noise only; real background noise, room impulse responses and device
  shifts are not yet evaluated.
- Fixed epochs avoid test-set model selection but are not a full inner-validation hyperparameter study.
- Reported model inference time excludes WAV loading, resampling and log-mel extraction.
- The four known cross-fold Freesound source IDs in official ESC-50 are retained for comparability and
  remain disclosed by manifest validation.
- This is a compact portfolio baseline, not a state-of-the-art ESC-50 claim.

## Reproduction

```bash
.venv/bin/python -m audioscope.cli cnn-cross-validate \
  --config configs/esc50_logmel_cnn.yaml
.venv/bin/python -m audioscope.cli cnn-cross-validate \
  --config configs/esc50_logmel_cnn_noise_augmented.yaml
```

Fold-level metrics, training curves, confusion matrices and error JSONL files are under
`reports/generated/esc50-logmel-cnn*/`. Models are under `artifacts/esc50-logmel-cnn*/`.

