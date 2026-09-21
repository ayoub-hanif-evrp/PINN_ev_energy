PYTHON ?= python

.PHONY: help install test audit smoke quick paper preprocess figures tables protocol

help:
	@echo "Targets:"
	@echo "  make install     Install package in editable mode"
	@echo "  make test        Run unit tests (synthetic fixtures)"
	@echo "  make audit       Discover HELECAR-D and write data audit"
	@echo "  make smoke       Short LOTO engineering run (not paper numbers)"
	@echo "  make quick       Full-trip LOTO diagnostic (not paper numbers)"
	@echo "  make protocol    Freeze paper protocol after quick debugging"
	@echo "  make paper       Frozen final pipeline (reuses cached identical runs)"
	@echo "  make figures     Publication figures from outputs/"
	@echo "  make tables      CSV/LaTeX tables from outputs/"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

audit:
	$(PYTHON) scripts/audit_data.py --config configs/base.yaml

preprocess:
	$(PYTHON) scripts/audit_data.py --config configs/base.yaml --preprocess

smoke:
	$(PYTHON) scripts/run_loto.py --config configs/smoke.yaml

quick:
	$(PYTHON) scripts/run_loto.py --config configs/quick.yaml

protocol:
	$(PYTHON) scripts/freeze_protocol.py --config configs/paper.yaml

figures:
	$(PYTHON) scripts/make_paper_figures.py --results outputs/ --config configs/paper.yaml

tables:
	$(PYTHON) scripts/make_paper_tables.py --config configs/paper.yaml

paper:
	$(PYTHON) scripts/audit_data.py --config configs/base.yaml
	$(PYTHON) scripts/freeze_protocol.py --config configs/paper.yaml
	$(PYTHON) scripts/run_loto.py --config configs/paper.yaml
	$(PYTHON) scripts/run_ablations.py --config configs/paper.yaml
	$(PYTHON) scripts/run_data_scarcity.py --config configs/paper.yaml
	$(PYTHON) scripts/run_cross_trajectory.py --config configs/paper.yaml
	$(PYTHON) scripts/run_sensitivity.py --config configs/paper.yaml
	$(PYTHON) scripts/run_feasibility_sensitivity.py --config configs/paper.yaml
	$(PYTHON) scripts/make_paper_figures.py --results outputs/ --config configs/paper.yaml
	$(PYTHON) scripts/make_paper_tables.py --config configs/paper.yaml
	$(PYTHON) scripts/export_paper_reports.py --config configs/paper.yaml
