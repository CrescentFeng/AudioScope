# AudioScope Polyphonic ESC-10 v1 dataset card

## Purpose

This generated dataset provides deterministic, strongly timed, overlapping sound events for the
first AudioScope sound-event-detection baseline. It is a controlled bridge from clip classification
to frame-level multilabel detection, not a substitute for evaluation on real strongly annotated
recordings.

## Construction

- Source: the ten ESC-10 classes inside the pinned local ESC-50 revision.
- Scene duration: 10 seconds at 16 kHz mono.
- Events per scene: 2-4, with event crops lasting 1-3 seconds.
- Splits: ESC folds 1-3 for train, fold 4 for validation, fold 5 for test.
- Leakage guard: any source recording ID occurring in more than one split is removed before mixing.
- Mixing: deterministic seed 2027, randomized crop, gain, onset and overlap; exact mixing intervals
  become onset/offset annotations.

| Split | Scenes | Events | Unique source recordings | Scenes with overlap |
|---|---:|---:|---:|---:|
| Train | 300 | 921 | 185 | 275 (91.7%) |
| Validation | 100 | 301 | 58 | 91 (91.0%) |
| Test | 100 | 293 | 55 | 88 (88.0%) |

Maximum observed polyphony is four. Source overlap checks between all three splits are empty. Two
ESC-10 source IDs that crossed the requested split boundary were excluded.

## Labels

`chainsaw`, `clock_tick`, `crackling_fire`, `crying_baby`, `dog`, `helicopter`, `rain`, `rooster`,
`sea_waves`, `sneezing`.

Each JSONL scene record stores the WAV path, split, duration, sample rate and a list of label/onset/
offset/source tuples. Generated audio and labels are reproducible from the source manifest and YAML
configuration.

## License and limitations

The dataset inherits ESC-50's CC BY-NC 3.0 license and is restricted to research and portfolio use.
ESC-50 labels are clip-level: treating the entire selected crop as active gives exact mixing-time
annotations but may overstate the true acoustic activity inside intermittent source clips. Reusing a
source within one split also means soundscapes are not fully independent. Scores must be described as
controlled synthetic-polyphony results, not real-world SED performance.
