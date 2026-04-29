.DEFAULT_GOAL := help

TRAIN_CONFIG ?= configs/experiments/yolov12n_cpu_smoke.yaml
EXPERIMENT_CONFIG ?= configs/experiments/yolov12n_cpu_100e_416_b2.yaml
MAX_SAMPLES ?= 20
LIMIT ?= 10

.PHONY: help sync test lint format format-check check \
        inspect train-dry train evaluate-dry evaluate \
        predict-dry predict report errors errors-sample app

help:
	@echo "Available commands:"
	@echo ""
	@echo "  make sync             Install project and development dependencies"
	@echo "  make test             Run pytest"
	@echo "  make lint             Run Ruff linting"
	@echo "  make format           Format code with Ruff"
	@echo "  make format-check     Check code formatting"
	@echo "  make check            Run linting, formatting check, and tests"
	@echo ""
	@echo "  make inspect          Inspect experiment config"
	@echo "  make train-dry        Validate training config without training"
	@echo "  make train            Run training from config"
	@echo "  make evaluate-dry     Validate evaluation config without evaluation"
	@echo "  make evaluate         Run evaluation from config"
	@echo "  make predict-dry      Validate prediction config without prediction"
	@echo "  make predict          Run prediction from config"
	@echo "  make report           Create prediction report"
	@echo "  make errors           Run full error analysis"
	@echo "  make errors-sample    Run error analysis on a small image subset"
	@echo "  make app              Launch Gradio app"
	@echo ""
	@echo "Config variables:"
	@echo "  TRAIN_CONFIG=$(TRAIN_CONFIG)"
	@echo "  EXPERIMENT_CONFIG=$(EXPERIMENT_CONFIG)"
	@echo ""
	@echo "Override examples:"
	@echo "  make train TRAIN_CONFIG=configs/experiments/yolov12n_cpu_smoke.yaml"
	@echo "  make evaluate EXPERIMENT_CONFIG=configs/experiments/yolov12n_cpu_100e_416_b2.yaml"

sync:
	uv sync --dev

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

check: lint format-check test

inspect:
	uv run python scripts/inspect_experiment_config.py \
		--config $(EXPERIMENT_CONFIG)

train-dry:
	uv run python scripts/train_yolov12.py \
		--config $(TRAIN_CONFIG) \
		--dry-run

train:
	uv run python scripts/train_yolov12.py \
		--config $(TRAIN_CONFIG)

evaluate-dry:
	uv run python scripts/evaluate_yolov12.py \
		--config $(EXPERIMENT_CONFIG) \
		--dry-run

evaluate:
	uv run python scripts/evaluate_yolov12.py \
		--config $(EXPERIMENT_CONFIG)

predict-dry:
	uv run python scripts/predict_yolov12.py \
		--config $(EXPERIMENT_CONFIG) \
		--dry-run

predict:
	uv run python scripts/predict_yolov12.py \
		--config $(EXPERIMENT_CONFIG)

report:
	uv run python scripts/create_prediction_report.py \
		--config $(EXPERIMENT_CONFIG) \
		--max-samples $(MAX_SAMPLES)

errors:
	uv run python scripts/analyze_yolov12_errors.py \
		--config $(EXPERIMENT_CONFIG)

errors-sample:
	uv run python scripts/analyze_yolov12_errors.py \
		--config $(EXPERIMENT_CONFIG) \
		--limit $(LIMIT)

app:
	uv run python scripts/run_gradio_app.py
