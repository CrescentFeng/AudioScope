# AudioScope L2 transfer-learning model card

## Scope

This stage evaluates frozen pretrained audio representations on ESC-50. It does not fine-tune the
AST or CLAP encoder. The exported trainable artifacts are multinomial logistic-regression probes;
CLAP also exports fixed text-prompt embeddings for zero-shot classification.

## Encoders

- `MIT/ast-finetuned-audioset-10-10-0.4593`, revision
  `f826b80d28226b62986cc218e5cec390b1096902`, BSD-3-Clause, 768-dimensional embedding.
- `laion/clap-htsat-unfused`, revision
  `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`, Apache-2.0, 512-dimensional embedding.

Exact weight hashes are registered in `data/sources.yaml`. ESC-50 remains CC BY-NC 3.0, so the
evaluation package is for research and portfolio use rather than unrestricted commercial use.

## Evaluation

All supervised results use the five official ESC-50 folds. Each fold fits the scaler and linear
probe on the other four folds only. CLAP zero-shot uses three fixed English templates per class and
does not fit on ESC-50 audio. AST sample efficiency uses a deterministic hash-ranked, class-balanced
subset of each training split.

| Route | Accuracy | Macro-F1 |
|---|---:|---:|
| CLAP zero-shot | 0.844 ± 0.014 | 0.831 ± 0.017 |
| AST linear probe | 0.961 ± 0.017 | 0.960 ± 0.019 |
| CLAP linear probe | 0.968 ± 0.007 | 0.967 ± 0.008 |

## Limitations

- AST is pretrained/fine-tuned on AudioSet and CLAP is trained on large audio-text corpora. ESC-50
  concepts overlap these pretraining ontologies, so the result measures transfer rather than learning
  all sound concepts from the 2,000 ESC-50 clips alone.
- The official ESC-50 protocol contains four source recording IDs that cross folds. They are retained
  for comparability and disclosed; the same caveat applies to L1 results.
- Zero-shot performance depends on English label wording. The prompts are fixed and stored, but a
  different prompt set can change the score.
- The task is closed-set, single-label, five-second classification. It does not establish unknown
  sound rejection, event timing, streaming latency, or real-device robustness.

## Intended use

Use these checkpoints to compare frozen-representation quality, bootstrap a labelled environmental
sound classifier, and support AudioScope L3 experiments. Do not present the scores as an open-world
sound-recognition guarantee.
