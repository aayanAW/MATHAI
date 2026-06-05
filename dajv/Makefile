# DAJV / verifyensemble — common dev tasks.
# Usage: `make help` for a list.

PYTHON := python3
PIP := pip
PYTHONPATH_PREFIX := PYTHONPATH=.

.PHONY: help install test lint type check paper experiments figures new-extractors clean

help:
	@echo "Targets:"
	@echo "  install      pip install -e ."
	@echo "  test         pytest"
	@echo "  lint         ruff check"
	@echo "  type         mypy"
	@echo "  check        lint + type + test"
	@echo "  paper        compile paper PDF (tectonic)"
	@echo "  experiments  run all empirical scripts (no API keys needed)"
	@echo "  figures      regenerate paper figures"
	@echo "  clean        remove caches + figures + artifacts"

install:
	$(PIP) install -e .
	$(PIP) install pytest hypothesis ruff mypy matplotlib

test:
	$(PYTHONPATH_PREFIX) $(PYTHON) -m pytest tests/ -q

lint:
	ruff check verifyensemble/ scripts/ tests/

type:
	mypy verifyensemble/

check: lint type test

paper:
	cd paper && tectonic main.tex

experiments:
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_dependency_mapping.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_synthetic_validation.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_aggregation_comparison.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_ablations.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_loo_ablation.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_seed_stability.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_prm_head_to_head.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_prm_matched_coverage.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_within_lab_dependency.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_solver_rotation.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_copula_ablation.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_threshold_sensitivity.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_hybrid_modality.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_hybrid_aggregation.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_verge_proxy_compare.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_h6_significance.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_modality_conjunctive_dajv.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_xmod_dajv.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_verge_z3_compare.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_8extractor_aggregation.py --min-records 300 --out aggregation_kfull.json
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_headline_metrics.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/example_quickstart.py

figures:
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_figures.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_risk_coverage_bands.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_hybrid_modality_figure.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_h6_figure.py
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_cross_modality_heatmap.py

new-extractors:
	@echo "Run scripts/run_new_extractors.py to collect E08S/E04S/E10S/E12 caches."
	@echo "Requires OPENAI_API_KEY and GOOGLE_API_KEY in .env."
	@echo "Together AI extractors (E02/E04/E10) require user-side credit top-up."
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_new_extractors.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis
	rm -f paper/main.pdf
