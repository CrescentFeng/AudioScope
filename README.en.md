# AudioScope

**Workbench enhancement:** replayable events, append-only corrections, aligned reference timelines,
model comparison, history/exports and local text retrieval. Use `python -m audioscope.launch` from the
configured environment, or run the Python containers with explicit host ports:

```bash
AUDIOSCOPE_API_PORT=8002 AUDIOSCOPE_DEMO_PORT=7862 docker compose --profile demo up -d --no-build api demo
```

Build first with `docker compose build api`. Workbench: `http://127.0.0.1:7862`; API documentation:
`http://127.0.0.1:8002/docs`. Local data and models are required and are not automatically downloaded.
See [usage](docs/WORKBENCH_GUIDE.md), [evidence and tradeoffs](reports/WORKBENCH_IMPROVEMENT.md), and
[enhancement release record](docs/WORKBENCH_RELEASE.md). The package remains 1.0.0; evidence hashes
identify this enhancement separately from the original release archive.

Validation-selected decoding improves clean event-F1 from 0.2583 to 0.2873 on the existing 221-clip test
split, but reduces recall; mean event-F1 across six controlled software shifts improves from 0.2210
to 0.2568. Reviews do not automatically enter training. English CLAP retrieval returns candidates,
including when the target sound is absent; similarity is not correctness probability.

Original release: **AudioScope v1.0 Software-Validated Release**. Physical device and room certification
has not been performed. The software gate covers seven deterministic domain shifts, bounded streaming
over 1,800 seconds of audio, and an automatically narrated evidence video. See `docs/SOFTWARE_RELEASE.md`.

AudioScope is a reproducible environmental-audio portfolio project covering classical and deep
classification, transfer learning, real strongly labelled sound event detection, short-event optimization,
streaming and long-form inference, open-set confidence rejection, ONNX CPU deployment, and a traceable data
catalog.

Measured highlights on the current Apple CPU are 0.873/0.967 ms P50/P95 model latency for a ten-second
ONNX FP32 window and a 0.000381 end-to-end real-time factor on an approximately 120-second file. The
open-set confidence baseline reaches 0.750 AUROC. These values are tied to the checked-in evaluation protocol;
dynamic INT8 did not accelerate this CRNN and is retained as a negative result.

## Reproduce the engineering package

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[engineering,demo,database,api,dev]'
.venv/bin/python -m pytest
.venv/bin/audioscope l4-build --config configs/l4_engineering.yaml
```

Database and Docker instructions are in `docs/DATABASE_AND_DOCKER.md`. Field preparation and hardware
acceptance are in `docs/FIELD_OPERATIONS.md`. The Chinese/English portfolio summary, resume bullets, and
video script are also under `docs/`.

Use `software-final` for this software release. `field-final` and its legacy alias `final` retain the
physical recording requirements; these are an optional future extension.

## License

AudioScope source code is released under the [MIT License](LICENSE). Third-party datasets, audio,
pretrained models, and derived assets remain subject to their respective source terms and licenses;
see `data/sources.yaml` before downloading or redistributing them.

## Rights and attribution / 权利与署名

Original project author/maintainer: **Crescent Feng (CrescentFeng)**. The current MIT licence retains copyright while permitting reuse under its terms, including commercial reuse; it is not an all-rights-reserved/no-commercial-use policy. Third-party material retains its own licence.

- [Copyright, attribution and contact](COPYRIGHT.md)
- [Citation](CITATION.cff) and [third-party notices](THIRD_PARTY_NOTICES.md)
- [Asset provenance](docs/ASSET_PROVENANCE.md)
- [Contributing](CONTRIBUTING.md) and [security reporting](SECURITY.md)
