.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install lint lint/flake8 \
	train train-modal train-damage train-damage-modal train-combined train-combined-modal \
	train-damage-shapenet train-damage-shapenet-modal train-combined-shapenet \
	train-combined-shapenet-modal shapenet-volume-create shapenet-volume-ls \
	shapenet-volume-upload shapenet-volume-download shapenet-volume-ls-data \
	shapenet-subset-download shapenet-damage-subset-download
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-32s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"
PYTHON ?= python
TRAIN_SCRIPT ?= scripts/train.py
SHAPENET_SUBSET_DOWNLOAD_SCRIPT ?= scripts/download_shapenet_subset.py
SHAPENET_DOWNLOAD_SCRIPT ?= scripts/download_shapenet_volume.py
CONFIG ?= configs/train_combined.yaml
MODAL_CONFIG ?= configs/train_combined_modal.yaml
DAMAGE_CONFIG ?= configs/train_damage.yaml
DAMAGE_MODAL_CONFIG ?= configs/train_damage_modal.yaml
COMBINED_CONFIG ?= configs/train_combined.yaml
COMBINED_MODAL_CONFIG ?= configs/train_combined_modal.yaml
DAMAGE_SHAPENET_CONFIG ?= configs/train_damage_shapenet.yaml
DAMAGE_SHAPENET_MODAL_CONFIG ?= configs/train_damage_shapenet_modal.yaml
COMBINED_SHAPENET_CONFIG ?= configs/train_shapenet.yaml
COMBINED_SHAPENET_MODAL_CONFIG ?= configs/train_shapenet_modal.yaml
SHAPENET_VOLUME ?= shapenet-voxels
SHAPENET_LOCAL_DIR ?= data/shapenet_voxels
SHAPENET_REMOTE_DIR ?= /
SHAPENET_REMOTE_PATH ?= /shapenet_voxels
SHAPENET_DOWNLOAD_DIR ?= data

help: ## show this help message
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr coverage/
	rm -fr .pytest_cache

lint: ## check style with flake8
	isort --profile black regen
	black regen
	flake8 regen

install: clean lint
	python -m pip install . --upgrade

doc:
	rm -r docs/reference/
	pdocs as_markdown regen -o docs/reference
	rm docs/reference/regen/index.md
	cp examples/*.ipynb docs/examples/
	cp README.md docs/index.md

serve-docs:
	mkdocs serve

commit: install test doc
	git add .
	git commit -a

test:
	python -m pytest --cov=regen/ --cov-report html:tests/cov-report tests/

test-html: test
	$(BROWSER) tests/cov-report/index.html

shapenet-volume-create: ## create the Modal ShapeNet volume if needed
	@output="$$(modal volume create $(SHAPENET_VOLUME) 2>&1)" || { \
		status=$$?; \
		echo "$$output"; \
		case "$$output" in \
			*"already exists"*) echo "Volume $(SHAPENET_VOLUME) already exists; continuing." ;; \
			*) exit $$status ;; \
		esac; \
	}; \
	if [ -n "$$output" ]; then echo "$$output"; fi

shapenet-volume-ls: ## list files in the Modal ShapeNet volume
	modal volume ls $(SHAPENET_VOLUME)

shapenet-volume-ls-data: ## list the uploaded ShapeNet data directory in the Modal volume
	modal volume ls $(SHAPENET_VOLUME) $(SHAPENET_REMOTE_PATH)

shapenet-volume-upload: ## upload SHAPENET_LOCAL_DIR into the Modal ShapeNet volume
	@if [ ! -d "$(SHAPENET_LOCAL_DIR)" ]; then \
		echo "Missing local ShapeNet voxel directory: $(SHAPENET_LOCAL_DIR)"; \
		echo "Run make shapenet-subset-download first, or set SHAPENET_LOCAL_DIR to an existing voxel directory."; \
		exit 1; \
	fi
	modal volume put --force $(SHAPENET_VOLUME) $(SHAPENET_LOCAL_DIR) $(SHAPENET_REMOTE_DIR)

shapenet-volume-download: ## download ShapeNet data from the Modal volume
	$(PYTHON) $(SHAPENET_DOWNLOAD_SCRIPT) --volume $(SHAPENET_VOLUME) --remote-path $(SHAPENET_REMOTE_PATH) --local-dir $(SHAPENET_DOWNLOAD_DIR)

shapenet-subset-download: ## download sampled ShapeNet files for combined training
	$(PYTHON) $(SHAPENET_SUBSET_DOWNLOAD_SCRIPT) \
		--config $(COMBINED_SHAPENET_MODAL_CONFIG) \
		--output-root $(SHAPENET_LOCAL_DIR) \
		--force

shapenet-damage-subset-download: ## download sampled ShapeNet files for damage training
	$(PYTHON) $(SHAPENET_SUBSET_DOWNLOAD_SCRIPT) \
		--config $(DAMAGE_SHAPENET_MODAL_CONFIG) \
		--output-root $(SHAPENET_LOCAL_DIR) \
		--force

train: ## train with CONFIG=configs/train_combined.yaml (override CONFIG=...)
	$(PYTHON) $(TRAIN_SCRIPT) --config $(CONFIG)

train-modal: ## train with MODAL_CONFIG=configs/train_combined_modal.yaml
	$(PYTHON) $(TRAIN_SCRIPT) --config $(MODAL_CONFIG)

train-damage: ## train damage direction detection locally
	$(PYTHON) $(TRAIN_SCRIPT) --config $(DAMAGE_CONFIG)

train-damage-modal: ## train damage direction detection on Modal
	$(PYTHON) $(TRAIN_SCRIPT) --config $(DAMAGE_MODAL_CONFIG)

train-combined: ## train combined class + damage detection locally
	$(PYTHON) $(TRAIN_SCRIPT) --config $(COMBINED_CONFIG)

train-combined-modal: ## train combined class + damage detection on Modal
	$(PYTHON) $(TRAIN_SCRIPT) --config $(COMBINED_MODAL_CONFIG)

train-damage-shapenet: ## train damage direction detection on ShapeNet locally
	$(PYTHON) $(TRAIN_SCRIPT) --config $(DAMAGE_SHAPENET_CONFIG)

train-damage-shapenet-modal: ## download/upload sampled ShapeNet subset, then train damage detection on Modal
	$(PYTHON) $(SHAPENET_SUBSET_DOWNLOAD_SCRIPT) --config $(DAMAGE_SHAPENET_MODAL_CONFIG) --output-root $(SHAPENET_LOCAL_DIR) --force
	$(MAKE) shapenet-volume-create
	$(MAKE) shapenet-volume-upload
	$(PYTHON) $(TRAIN_SCRIPT) --config $(DAMAGE_SHAPENET_MODAL_CONFIG)

train-combined-shapenet: ## train combined class + damage detection on ShapeNet locally
	$(PYTHON) $(TRAIN_SCRIPT) --config $(COMBINED_SHAPENET_CONFIG)

train-combined-shapenet-modal: ## download/upload sampled ShapeNet subset, then train combined model on Modal
	$(PYTHON) $(SHAPENET_SUBSET_DOWNLOAD_SCRIPT) --config $(COMBINED_SHAPENET_MODAL_CONFIG) --output-root $(SHAPENET_LOCAL_DIR) --force
	$(MAKE) shapenet-volume-create
	$(MAKE) shapenet-volume-upload
	$(PYTHON) $(TRAIN_SCRIPT) --config $(COMBINED_SHAPENET_MODAL_CONFIG)
