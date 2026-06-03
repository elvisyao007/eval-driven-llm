# Reproducible entry points. Each target is the exact command a run uses.

.PHONY: install install-base test build-toy eval-toy build-jqara eval ingest

install-base:          # deterministic baseline loop, no model downloads
	pip install -e ".[dev,base]"

install:               # full production semantic stack (downloads weights)
	pip install -e ".[dev,track-a]"

test:
	pytest -q

# --- plumbing validation: tiny synthetic set, lexical baseline ---
build-toy:
	python scripts/build_toy_golden.py
eval-toy:
	python -m elv.eval.runner --golden data/golden/_toy/v0 --embedder hashing --ks 3,5
compare-toy:           # dense-only vs dense+lexical-rerank (plumbing demo)
	python -m elv.eval.runner --golden data/golden/_toy/v0 --embedder hashing --compare --rerank lexical --ks 3,5
rag-eval-toy:          # end-to-end faithfulness, OFFLINE self-test (no model)
	python -m elv.eval.rag_eval --golden data/golden/_toy/v0 --embedder hashing --gen template --judge test

# --- Track A: JQaRA, real semantic embedder (needs GPU + HF access) ---
build-jqara:           # freeze a JQaRA split into a golden set + corpus
	python -c "from elv.eval.adapters.jqara import build_frozen_set as b; b('test','v0','.')"
eval:                  # GOLDEN=data/golden/jqara/v0
	python -m elv.eval.runner --golden $(GOLDEN) --embedder ruri --ks 5,10
compare:               # the real one: dense-only vs dense+cross-encoder on JQaRA
	python -m elv.eval.runner --golden $(GOLDEN) --embedder ruri --compare --rerank cross-encoder --ks 5,10
rag-eval:              # end-to-end faithfulness with a pinned LOCAL judge
	python -m elv.eval.rag_eval --golden $(GOLDEN) --embedder ruri --rerank cross-encoder \
		--gen openai --judge local --base-url $(BASE_URL) --model $(GEN_MODEL) --judge-model $(JUDGE_MODEL)

# --- dirty-data ingestion (Track B): docs -> passages.jsonl + audit.json ---
build-dirty-fixtures:
	python scripts/build_dirty_fixtures.py
ingest-dirty: build-dirty-fixtures
	python -m elv.ingest.loader data/corpus/_dirty --out data/corpus/_ingested/passages.jsonl --owner team-a

# ingest any corpus dir:  make ingest CORPUS=path/to/docs OWNER=team-a
ingest:
	python -m elv.ingest.loader $(CORPUS) --owner $(OWNER)
