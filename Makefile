# OnnesSim — developer Makefile
# All targets use the project virtualenv interpreter so behavior matches CI.
# Usage: `make install`, `make test`, `make cooldown`, `make dataset`, `make eval`.

# Project-local venv python. Override on the CLI, e.g. `make test PY=python3`.
PY ?= .venv/bin/python

# Tunable script arguments (override on the CLI, e.g. `make dataset N=120 HOURS=24`).
FAULT    ?= normal
HOURS    ?= 36
N        ?= 60
DATASET  ?= outputs/dataset
BACKEND  ?= auto

.DEFAULT_GOAL := help
.PHONY: help install test cooldown dataset eval clean

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Editable install with dev+ml+llm extras (test/baseline/agent deps).
	$(PY) -m pip install -e ".[dev,ml,llm]"

test: ## Run the pytest suite.
	$(PY) -m pytest -q

cooldown: ## Run one cooldown -> outputs/cooldown.csv + .png (FAULT=, HOURS=).
	$(PY) scripts/run_cooldown.py --fault $(FAULT) --hours $(HOURS)

dataset: ## Generate a labeled scenario dataset (N=, HOURS=, out=DATASET).
	$(PY) scripts/generate_dataset.py --n $(N) --hours $(HOURS) --out $(DATASET)

eval: ## Evaluate agent vs threshold baseline on the dataset (BACKEND=, DATASET=).
	$(PY) scripts/evaluate.py --dataset $(DATASET) --backend $(BACKEND)

clean: ## Remove caches (leaves outputs/ untouched).
	rm -rf .pytest_cache */__pycache__ __pycache__
