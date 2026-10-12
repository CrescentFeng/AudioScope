# ESC-50 MFCC + SVM baseline

Experiment date: 2026-09-15  
Dataset revision: `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`  
Dataset license: CC BY-NC 3.0  
Runtime: Python 3.12.14, scikit-learn 1.9.1, Apple silicon macOS

## Protocol

- 2,000 five-second clips, 50 balanced classes, official five folds.
- Audio is converted to mono and resampled from 44.1 kHz to 32 kHz.
- Each clip is represented by the mean and standard deviation of 40 MFCC coefficients.
- The classifier is a standardized RBF SVM (`C=10`) with three-fold sigmoid calibration.
- Each official fold is used once as test data. No test fold is used for fitting or calibration.
- Robustness uses deterministic additive white noise at 20, 10 and 0 dB SNR.

## Results

| Test fold | Accuracy | Macro-F1 |
|---:|---:|---:|
| 1 | 0.360 | 0.292 |
| 2 | 0.375 | 0.315 |
| 3 | 0.388 | 0.330 |
| 4 | 0.418 | 0.351 |
| 5 | 0.325 | 0.274 |
| **Mean ± population SD** | **0.373 ± 0.031** | **0.312 ± 0.027** |

The five fitted SVMs took 1.80 seconds in total after feature extraction. The recorded prediction
time (0.25 seconds for all 2,000 test decisions) covers only classifier inference; it excludes WAV
loading, resampling and MFCC extraction and therefore must not be presented as end-to-end latency.

## White-noise robustness

| SNR | Mean Accuracy | Mean Macro-F1 |
|---:|---:|---:|
| Clean | 0.373 | 0.312 |
| 20 dB | 0.268 | 0.225 |
| 10 dB | 0.171 | 0.130 |
| 0 dB | 0.061 | 0.032 |

The sharp drop at 20 dB is the clearest baseline failure. The CNN stage should include controlled
noise augmentation and must be evaluated using the same noise generator before adding real-noise and
room-impulse-response tests.

## Error analysis

The four lowest-recall classes (`door_wood_creaks`, `laughing`, `mouse_click`, `water_drops`) had zero
correct predictions across their 40 test clips. Frequent directional confusions included:

- `fireworks → thunderstorm`: 16
- `siren → clock_alarm`: 16
- `coughing → sneezing`: 13
- `airplane → thunderstorm`: 12
- `cat → crying_baby`: 11
- `church_bells → clock_alarm`: 11

EDA found 552 clips whose decoded peak reaches full scale. These are review candidates, not automatic
proof of audible clipping. The source audio is preserved without normalization.

## Official-fold caveat

The official metadata places four Freesound source IDs in more than one fold: `131943`, `134049`,
`209698` and `234879`. They correspond to different ESC-50 target classes (for example, clock alarm
versus clock tick) but still share source recordings. This run preserves the official folds for
comparability and records the overlap in manifest validation. A separate source-disjoint sensitivity
analysis excludes all nine affected clips and evaluates the remaining 1,991 records. It produces
Accuracy `0.374 ± 0.028` and Macro-F1 `0.316 ± 0.022`, close to the official-fold result. This supports
the conclusion that the four overlaps are not driving the headline score; the official result remains
the comparable headline.

## Reproduction

```bash
.venv/bin/python -m audioscope.cli prepare-esc50 \
  --dataset-root data/raw/ESC-50-master \
  --output data/manifests/esc50.jsonl
.venv/bin/python -m audioscope.cli cross-validate \
  --config configs/esc50_mfcc_svm.yaml
.venv/bin/python -m audioscope.cli cross-validate \
  --config configs/esc50_mfcc_svm_source_disjoint.yaml
```

Generated fold-level metrics, confusion matrices and error JSONL files are stored under
`reports/generated/esc50-mfcc-svm/`. Model artifacts and cached features are excluded from Git.
