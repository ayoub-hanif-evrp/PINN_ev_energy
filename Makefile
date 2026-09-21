PYTHON ?= python

.PHONY: help install audit paper figures tables

help:
	@echo "Targets:"
	@echo "  make install   Install package in editable mode"
	@echo "  make audit     Discover HELECAR-D and write a data audit"
	@echo "  make paper     Final pipeline (reuses existing LOTO outputs)"
	@echo "  make figures   PNG figures under results/figures/"
	@echo "  make tables    CSV tables under results/tables/"

install:
	$(PYTHON) -m pip install -e .

audit:
	$(PYTHON) scripts/audit_data.py --config configs/base.yaml

figures:
	$(PYTHON) scripts/make_final_figures.py --config configs/paper.yaml --out results

tables:
	$(PYTHON) scripts/make_final_tables.py --config configs/paper.yaml --out results

paper:
	$(PYTHON) scripts/run_final_pipeline.py --config configs/paper.yaml
