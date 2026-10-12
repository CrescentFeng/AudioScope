# Pretrained model registry

L2 uses frozen upstream encoders. Large upstream weights are ignored by Git; their identity is fixed
by revision and SHA-256 in `sources.yaml`.

## AST

- Model: `MIT/ast-finetuned-audioset-10-10-0.4593`
- Revision: `f826b80d28226b62986cc218e5cec390b1096902`
- Weight: `model.safetensors`, 346,404,948 bytes
- SHA-256: `ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d`
- Local source used: Hugging Face cache at the pinned snapshot

## CLAP

- Model: `laion/clap-htsat-unfused`
- Revision: `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`
- Weight: `pytorch_model.bin`, 614,525,833 bytes
- SHA-256: `1cd3c601bc4afe0fa87be3de4c13dd2cfadd249fac1e29acf74a9b296c3219bb`
- Local path: `data/pretrained/clap-htsat-unfused-8fa0f1c/`

The CLAP config intentionally uses the local snapshot and `local_files_only: true`, so evaluation
does not silently switch revisions. To recreate it, download the files from the pinned model revision
into the path above and verify the weight hash before running `make esc50-clap`.
