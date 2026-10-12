# DESED real strong subset dataset card

## Source

Zenodo record: `https://doi.org/10.5281/zenodo.14808180`, “DESED in WebDataset Format” v2. The
record declares CC BY 4.0. All 40 downloaded shard MD5 values match the record metadata.

## Selection

The repack contains a 1578-clip weakly labelled train pool and an 1153-clip strongly annotated eval
pool. AudioScope excludes the weak pool from this experiment and divides only the strong pool using
a stable SHA-256 hash of the source YouTube video ID.

| Split | Clips | Events | Events ≤1 s |
|---|---:|---:|---:|
| Train | 703 | 2621 | 1264 |
| Validation | 229 | 826 | 398 |
| Test | 221 | 777 | 361 |

There is no source video ID overlap between splits. The manifest stores label, onset, offset, source
asset ID and source recording ID for every event.

## Intended use

Reproducible research and portfolio evaluation of supervised sound event detection, thresholding and
short-event behaviour. This split is not an official DCASE leaderboard protocol.

## Limitations

The class distribution is highly imbalanced and the underlying acoustic material originated from
AudioSet/YouTube. Users must verify underlying content terms for their own redistribution or
commercial scenario rather than relying only on the repack-level license declaration.
