# AudioScope L3 enhanced SED model card

## Recommended artifact

`artifacts/desed-real-enhanced/crnn_short_event.pt`

The model is a 246,762-parameter log-mel CRNN trained on 703 real recordings from the strongly
annotated DESED evaluation pool. Source video IDs are disjoint across train, validation and test.

## Threshold profiles

- `overall`: validation-selected global threshold 0.8; test event-F1 0.258 and segment-F1 0.593.
- `short_event`: validation-selected threshold 0.9 for events up to one second; test short-event F1
  0.210 and overall event-F1 0.274.
- `class_balanced`: ten validation-selected class thresholds. This is retained for analysis but is
  not recommended as the default because it reduced test micro and macro event-F1 for this model.

## Training intervention

Positive frames in events no longer than one second receive 2.5x loss weight. Frames within 80 ms
of an onset or offset receive at least 1.5x weight. Compared with the same CRNN without this
intervention, short-event recall rose from 0.205 to 0.360 at the overall profile, while precision
fell. The short-event threshold profile is required to turn this recall shift into an F1 gain.

## Comparison

The 184,362-parameter Lite Conformer reached test event-F1 0.221 and segment-F1 0.567, below the
enhanced CRNN, while taking 592 seconds versus 183 seconds to train on CPU. This comparison applies
only to the included small, randomly initialized models.

## Limitations

The source is a third-party WebDataset repack of DESED/AudioSet material. The experiment uses one
deterministic source-group split, has strong class imbalance, and is not calibrated for safety or
production monitoring. Inference timing excludes log-mel extraction.
