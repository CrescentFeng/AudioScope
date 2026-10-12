# Model card - LogMel CNN

Status: L1 CNN baseline complete; not a production model.

The model is a 36,370-parameter 2D convolutional classifier for the 50 ESC-50 classes. It consumes
five-second mono clips resampled to 32 kHz and represented as `64 × 250` log-mel images. It was
evaluated with the official five folds and fixed training settings.

The clean model achieved Accuracy `0.505 ± 0.041` and Macro-F1 `0.480 ± 0.041`. A controlled
waveform-noise training variant achieved clean Macro-F1 `0.468 ± 0.033`, 20 dB Macro-F1 `0.416`,
10 dB Macro-F1 `0.323` and 0 dB Macro-F1 `0.101`.

Use is limited to research and portfolio evaluation because ESC-50 is CC BY-NC 3.0. The model must
not be described as robust to real devices, rooms or unknown classes. Microphone deployment, open-set
rejection, calibration quality and end-to-end latency are not yet validated.

