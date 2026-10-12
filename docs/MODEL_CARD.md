# Model card - MFCC + SVM baseline

Status: first reportable traditional baseline complete; not a production model.

The checked-in configuration exercises data generation, manifest validation, feature extraction,
training, evaluation, robustness measurement and report export. Its synthetic tones are trivially
separable and are not evidence of environmental-sound performance.

The MFCC + calibrated SVM baseline was evaluated on all five official ESC-50 folds. It achieved
0.373 ± 0.031 Accuracy and 0.312 ± 0.027 Macro-F1. Mean Macro-F1 under deterministic white noise was
0.225 at 20 dB, 0.130 at 10 dB and 0.032 at 0 dB. These results establish a deliberately lightweight
floor for the CNN and pretrained-model comparisons.

Known limitations include weak transient modelling, severe noise sensitivity, no device or room
generalisation test, and no open-set rejection. Four Freesound source IDs appear in more than one
official fold; the official protocol is retained for comparability and the overlap is disclosed in the
baseline report. ESC-50 is non-commercial; downstream use must respect its CC BY-NC 3.0 terms.
