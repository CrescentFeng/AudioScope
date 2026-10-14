# Third-party notices / 第三方权利说明

AudioScope's root licence does not grant rights in third-party datasets,
recordings, model weights or other separately licensed material. Source
registrations and pinned revisions are maintained in [data/sources.yaml](data/sources.yaml).
The applicable upstream terms govern; a project metadata label cannot override them.

| Material | Source | Recorded terms and publication boundary |
| --- | --- | --- |
| ESC-50 | [karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50), revision `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6` | Recorded as CC-BY-NC-3.0; raw audio is not distributed here. Retain source identities and check the upstream terms before reuse. No commercial audio/training permission is inferred. |
| ESC-10 mixtures | Project recipes using ESC source clips | Derived audio is excluded; project code licensing does not remove underlying recording restrictions. |
| DESED WebDataset v2 | [Zenodo record 14808180](https://doi.org/10.5281/zenodo.14808180) | Repack declares CC-BY-4.0; underlying AudioSet/YouTube content rights require separate verification. No underlying audio is bundled or commercially cleared here. |
| AST | [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593), revision `f826b80d28226b62986cc218e5cec390b1096902` | Source register records BSD-3-Clause; weights are acquired separately. Review the pinned model's own terms and notices. |
| CLAP | [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused), revision `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a` | Source register records Apache-2.0; weights are not bundled. |
| External pilot recordings | `data/external-real-pilot-v1/selection.json` and `scripts/collect_external_pilot.py` | Each item keeps its creator, source, licence and changes. The original media is excluded. Aggregate project licensing does not replace per-item permissions. |
| Deterministic smoke fixtures | `src/audioscope/synthetic.py`, `data/sources.yaml` | Generated test data retains its existing CC0-1.0 declaration. This does not dedicate project code or third-party audio to the public domain. |

Runtime libraries and tools (including NumPy, SciPy, scikit-learn, PyTorch,
Transformers, ONNX, Gradio, FastAPI, SQLAlchemy, PostgreSQL and Docker images)
retain their own licences and notices. See `pyproject.toml` and the installed
packages/images for exact versions and transitive licences. No exhaustive
transitive licence audit is claimed by this inventory.

## Publication and attribution

The repository publishes source, configuration, manuals and experiment records.
It excludes raw recordings, pretrained weights, training checkpoints, local
user uploads and databases. Dataset names and model names identify sources;
they do not imply affiliation, endorsement or ownership by Crescent Feng.

Metadata, hashes and experimental scores are not permission to redistribute
underlying assets. When adding a new dependency or copying upstream code,
retain its licence and notices and identify modifications before publication.
