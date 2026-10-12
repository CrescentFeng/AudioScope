PYTHON ?= .venv/bin/python

.PHONY: setup test lint smoke cnn-smoke esc50-crossval esc50-cnn esc50-cnn-robust esc50-ast esc50-clap esc10-sed desed-enhanced demo db-up db-migrate db-import api-up field-prepare field-qc mic-sim release-preflight release-final clean-generated

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e '.[dev,demo]'

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests app.py

smoke:
	$(PYTHON) -m audioscope.cli smoke --config configs/smoke.yaml

cnn-smoke:
	$(PYTHON) -m audioscope.cli cnn-cross-validate --config configs/cnn_smoke.yaml

esc50-crossval:
	$(PYTHON) -m audioscope.cli cross-validate --config configs/esc50_mfcc_svm.yaml

esc50-cnn:
	$(PYTHON) -m audioscope.cli cnn-cross-validate --config configs/esc50_logmel_cnn.yaml

esc50-cnn-robust:
	$(PYTHON) -m audioscope.cli cnn-cross-validate --config configs/esc50_logmel_cnn_noise_augmented.yaml

esc50-ast:
	$(PYTHON) -m audioscope.cli transfer-evaluate --config configs/esc50_ast_linear_probe.yaml

esc50-clap:
	$(PYTHON) -m audioscope.cli transfer-evaluate --config configs/esc50_clap.yaml

esc10-sed:
	$(PYTHON) -m audioscope.cli sed-train-evaluate --config configs/esc10_polyphonic_crnn.yaml

desed-enhanced:
	$(PYTHON) -m audioscope.cli sed-enhance-evaluate --config configs/desed_real_enhanced.yaml

demo:
	$(PYTHON) app.py --config configs/smoke.yaml --model artifacts/smoke/model.joblib

db-up:
	docker compose up -d db

db-migrate:
	docker compose run --rm migrate

db-import:
	$(PYTHON) -m audioscope.catalog_import --root . --hash-files --output reports/generated/database/import_report.json

api-up:
	docker compose up -d api

field-prepare:
	$(PYTHON) -m audioscope.field_ops prepare --root . --clips 20

field-qc:
	$(PYTHON) -m audioscope.field_ops qc --root . --model artifacts/l4-engineering/crnn_short_event_calibrated.pt

mic-sim:
	$(PYTHON) -m audioscope.mic_acceptance --model artifacts/l4-engineering/crnn_short_event_calibrated.pt --onnx artifacts/l4-engineering/crnn_short_event_fp32.onnx --simulate-file data/l4-engineering/l4_long_demo.wav --output reports/generated/field/mic_simulation.json

release-preflight:
	$(PYTHON) -m audioscope.release_audit --root . --mode preflight

release-final:
	$(PYTHON) -m audioscope.release_audit --root . --mode final

software-evaluate:
	$(PYTHON) -m audioscope.software_validation evaluate

software-soak:
	$(PYTHON) -m audioscope.software_validation soak --seconds 1800

software-video:
	$(PYTHON) -m audioscope.release_video

software-final:
	$(PYTHON) scripts/verify_software_release.py
	$(PYTHON) -m audioscope.release_catalog sync
	$(PYTHON) -m audioscope.release_audit --mode software-final
	$(PYTHON) -m audioscope.release_catalog sync

field-final:
	$(PYTHON) -m audioscope.release_audit --mode field-final

clean-generated:
	$(PYTHON) -m audioscope.cli clean-generated --config configs/smoke.yaml
