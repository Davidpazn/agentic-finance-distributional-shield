# Execution-spine ablation (B0--B3) for the agentic-finance distributional shield.
#
# PYTHON can be overridden, e.g.  make reproduce PYTHON=.venv/bin/python
# Defaults to a local .venv if present, otherwise python3.
PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

.PHONY: reproduce spine test venv clean help

help:
	@echo "make venv       - create .venv and install requirements"
	@echo "make reproduce  - regenerate the execution-spine CSVs into data/ and verify the book table"
	@echo "make spine      - regenerate the CSVs without the verification gate"
	@echo "make test       - run the execution-spine assertions"
	@echo "make clean      - remove generated execution-spine CSVs"

venv:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt

reproduce:
	$(PYTHON) experiments/execution_spine.py --outdir data --check

spine:
	$(PYTHON) experiments/execution_spine.py --outdir data

test:
	$(PYTHON) -m pytest tests/test_execution_spine.py -q

clean:
	rm -f data/execution_spine_headline.csv \
	      data/execution_spine_metrics.csv \
	      data/execution_spine_audit_log.csv
