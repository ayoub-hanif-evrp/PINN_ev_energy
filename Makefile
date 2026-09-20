PYTHON ?= python

.PHONY: help install test audit paper preprocess

help:
	@echo "Targets:"
	@echo "  make install     Install package in editable mode"
	@echo "  make test        Run unit tests (synthetic fixtures)"
	@echo "  make audit       Discover HELECAR-D and write data audit"
	@echo "  make preprocess  Preprocess trips and write diagnostic plots"
	@echo "  make paper       Full paper pipeline (Phases 5+ not yet implemented)"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

audit:
	$(PYTHON) scripts/audit_data.py --config configs/base.yaml

preprocess:
	$(PYTHON) scripts/audit_data.py --config configs/base.yaml --preprocess

paper:
	$(PYTHON) scripts/run_loto.py --config configs/paper.yaml
	$(PYTHON) scripts/run_ablations.py --config configs/paper.yaml
	$(PYTHON) scripts/run_data_scarcity.py --config configs/paper.yaml
	$(PYTHON) scripts/run_sensitivity.py --config configs/paper.yaml
	$(PYTHON) scripts/run_cross_trajectory.py --config configs/paper.yaml
	$(PYTHON) scripts/make_paper_figures.py --results outputs/
