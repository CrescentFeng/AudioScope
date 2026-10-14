# Asset provenance / 素材来源

Reviewed for public documentation: 2026-10-14.

## Source and evidence

The repository records project source, original configuration, experiment
reports, a user manual and compact machine-readable release evidence. Commit
history attributes the current imported project to CrescentFeng; this alone
is not independent proof of ownership of every included item.

Third-party dataset/model identities and licences are listed in
[data/sources.yaml](../data/sources.yaml) and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
No WAV/MP3, pretrained model or trained checkpoint payload was found in the
reachable Git objects inspected for this publication. This does not clear
external data or weights for redistribution.

The user manual contains interface/flow illustrations and describes local
operation. The recorded software-validation results are historical evidence;
this documentation update does not rerun model evaluation or certify physical
rooms/devices. See [release scope](GITHUB_RELEASE.md).

## Data and generated fixtures

Dataset clips and mixtures stay outside Git. Metadata records their origins;
it is not a grant over the underlying sound. Per-item creator/licence records
for the external pilot remain in `data/external-real-pilot-v1/selection.json`.
Generated synthetic smoke data retains the existing CC0 declaration; it is
only a controlled test fixture, not evidence of real-world model accuracy.

For new assets, record creator, source URL, licence/version, distribution
permission, transformations and content hashes. Keep personal data and private
permission records outside Git. Preserve historical third-party terms and
previously granted permissions when revising the project licence.
