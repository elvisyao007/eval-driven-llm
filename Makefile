# Reproducible entry points. Each target is the exact command a run uses.

.PHONY: install install-base test build-toy eval-toy build-jqara eval ingest

VENV     := .venv
PIP      := $(VENV)/bin/pip
PYTHON   := $(VENV)/bin/python
PYTEST   := $(VENV)/bin/pytest

$(VENV)/bin/activate:
	python3 -m venv $(VENV)

install-base: $(VENV)/bin/activate   # deterministic baseline loop, no model downloads
	$(PIP) install -e ".[dev,base]"

install: $(VENV)/bin/activate        # full production semantic stack (downloads weights)
	$(PIP) install -e ".[dev,track-a]"

test: $(VENV)/bin/activate
	$(PYTEST) -q

# --- plumbing validation: tiny synthetic set, lexical baseline ---
build-toy:
	$(PYTHON) scripts/build_toy_golden.py
eval-toy:
	$(PYTHON) -m elv.eval.runner --golden data/golden/_toy/v0 --embedder hashing --ks 3,5
compare-toy:           # dense-only vs dense+lexical-rerank (plumbing demo)
	$(PYTHON) -m elv.eval.runner --golden data/golden/_toy/v0 --embedder hashing --compare --rerank lexical --ks 3,5
rag-eval-toy:          # end-to-end faithfulness, OFFLINE self-test (no model)
	$(PYTHON) -m elv.eval.rag_eval --golden data/golden/_toy/v0 --embedder hashing --gen template --judge test

# --- Track A: JQaRA, real semantic embedder (needs GPU + HF access) ---
build-jqara:           # freeze a JQaRA split into a golden set + corpus
	$(PYTHON) -c "from elv.eval.adapters.jqara import build_frozen_set as b; b('test','v0','.')"
eval:                  # GOLDEN=data/golden/jqara/v0
	$(PYTHON) -m elv.eval.runner --golden $(GOLDEN) --embedder ruri --ks 5,10
compare:               # the real one: dense-only vs dense+cross-encoder on JQaRA
	$(PYTHON) -m elv.eval.runner --golden $(GOLDEN) --embedder ruri --compare --rerank cross-encoder --ks 5,10
rag-eval:              # end-to-end faithfulness with a pinned LOCAL judge
	$(PYTHON) -m elv.eval.rag_eval --golden $(GOLDEN) --embedder ruri --rerank cross-encoder \
		--gen openai --judge local --base-url $(BASE_URL) --model $(GEN_MODEL) --judge-model $(JUDGE_MODEL)

# --- dirty-data ingestion (Track B): docs -> passages.jsonl + audit.json ---
build-dirty-fixtures:
	$(PYTHON) scripts/build_dirty_fixtures.py
ingest-dirty: build-dirty-fixtures
	$(PYTHON) -m elv.ingest.loader data/corpus/_dirty --out data/corpus/_ingested/passages.jsonl --owner team-a

# ingest any corpus dir:  make ingest CORPUS=path/to/docs OWNER=team-a
ingest:
	$(PYTHON) -m elv.ingest.loader $(CORPUS) --owner $(OWNER)
