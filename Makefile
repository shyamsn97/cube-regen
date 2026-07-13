.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install lint lint/flake8 \
	train train-modal train-damage train-damage-modal \
	train-damage-shape-conditioned train-damage-shape-conditioned-modal \
	train-combined train-combined-modal \
	train-damage-shapenet train-damage-shapenet-modal train-combined-shapenet \
	train-combined-shapenet-modal shapenet-volume-create shapenet-volume-ls \
	shapenet-volume-upload shapenet-volume-download shapenet-volume-ls-data \
	shapenet-subset-download shapenet-damage-subset-download \
	visualize-shapenet-predictions visualize-damage-prediction \
	visualize-shape-conditioned-recovery-gif visualize-shape-conditioned-seed-recovery-gif \
	visualize-shapenet-recovery-gif visualize-shapenet-seed-recovery-gif \
	visualize-combined-recovery-gif visualize-combined-seed-recovery-gif \
	visualize-sakana-damage visualize-sakana-recovery-gif \
	visualize-sakana-damage-recovery-gif visualize-sakana-seed-recovery-gif \
	train-sakana train-sakana-modal
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
INFERENCE_SCRIPT ?= scripts/inference.py
SHAPENET_RECOVERY_SCRIPT ?= scripts/recover_shapenet.py
COMBINED_RECOVERY_SCRIPT ?= scripts/recover_combined.py
SHAPE_CONDITIONED_RECOVERY_SCRIPT ?= scripts/recover_shape_conditioned.py
CONFIG ?= configs/train_combined.yaml
MODAL_CONFIG ?= configs/train_combined_modal.yaml
DAMAGE_CONFIG ?= configs/train_damage.yaml
DAMAGE_MODAL_CONFIG ?= configs/train_damage_modal.yaml
DAMAGE_SHAPE_CONDITIONED_CONFIG ?= configs/train_damage_shape_conditioned.yaml
DAMAGE_SHAPE_CONDITIONED_MODAL_CONFIG ?= configs/train_damage_shape_conditioned_modal.yaml
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
SHAPENET_PREDICTION_REPO ?= shyamsn97/shapenet-cube-regen-combined-hdim-48
SHAPENET_PREDICTION_WEIGHTS ?= pytorch_model.pt
SHAPENET_PREDICTION_OUTPUT ?= examples/shapenet_predictions
SHAPENET_PREDICTION_SAMPLES ?= 10
SHAPENET_RECOVERY_REPO ?= $(SHAPENET_PREDICTION_REPO)
SHAPENET_RECOVERY_WEIGHTS ?= $(SHAPENET_PREDICTION_WEIGHTS)
SHAPENET_RECOVERY_CONFIG ?= $(COMBINED_SHAPENET_MODAL_CONFIG)
SHAPENET_RECOVERY_OUTPUT_DIR ?= examples/recovery/shapenet
SHAPENET_RECOVERY_OUTPUT ?= shapenet_recovery.gif
SHAPENET_SEED_RECOVERY_OUTPUT ?= shapenet_seed_recovery.gif
SHAPENET_RECOVERY_CATEGORY ?=
SHAPENET_RECOVERY_SAMPLE_INDEX ?= 2
SHAPE_CONDITIONED_RECOVERY_REPO ?= shyamsn97/cube-regen-shape-conditioned-damage-hdim-64
SHAPE_CONDITIONED_RECOVERY_CONFIG ?= $(DAMAGE_SHAPE_CONDITIONED_MODAL_CONFIG)
SHAPE_CONDITIONED_RECOVERY_OUTPUT_DIR ?= examples/recovery/shape_conditioned
SHAPE_CONDITIONED_RECOVERY_OUTPUT ?= shape_conditioned_recovery.gif
SHAPE_CONDITIONED_SEED_RECOVERY_OUTPUT ?= shape_conditioned_seed_recovery.gif
SHAPE_CONDITIONED_RECOVERY_CLASS_NAME ?=
SHAPE_CONDITIONED_RECOVERY_CLASS_LABEL ?= 3
SHAPE_CONDITIONED_RECOVERY_SHAPE_SEED ?= 
SHAPE_CONDITIONED_RECOVERY_SAMPLE_INDEX ?= 0
SHAPE_CONDITIONED_RECOVERY_DEVICE ?= cpu
SHAPE_CONDITIONED_USE_ADAPTIVE_POOLING ?= false
SHAPE_CONDITIONED_RECOVERY_DAMAGE_TYPES ?= sphere cube
SHAPE_CONDITIONED_RECOVERY_DAMAGE_RADIUS_RANGE ?= 1 3
SHAPE_CONDITIONED_RECOVERY_NUM_DAMAGE_SITES_RANGE ?= 2 3
SHAPE_CONDITIONED_RECOVERY_MIN_DAMAGE_TARGET_CELLS ?= 8
SHAPE_CONDITIONED_RECOVERY_MAX_DAMAGE_ATTEMPTS ?= 10
COMBINED_RECOVERY_REPO ?= shyamsn97/cube-regen-combined-hdim-20
COMBINED_RECOVERY_CONFIG ?= $(COMBINED_MODAL_CONFIG)
COMBINED_RECOVERY_OUTPUT_DIR ?= examples/recovery/combined
COMBINED_RECOVERY_OUTPUT ?= combined_recovery.gif
COMBINED_SEED_RECOVERY_OUTPUT ?= combined_seed_recovery.gif
COMBINED_RECOVERY_CLASS_NAME ?=
COMBINED_RECOVERY_CLASS_LABEL ?= 3
COMBINED_RECOVERY_SHAPE_SEED ?= 0
COMBINED_RECOVERY_SAMPLE_INDEX ?= 0
COMBINED_RECOVERY_DAMAGE_TYPES ?= sphere cube
COMBINED_RECOVERY_DAMAGE_RADIUS_RANGE ?= 1 3
COMBINED_RECOVERY_NUM_DAMAGE_SITES_RANGE ?= 2 3
COMBINED_RECOVERY_MIN_DAMAGE_TARGET_CELLS ?= 8
COMBINED_RECOVERY_MAX_DAMAGE_ATTEMPTS ?= 10
COMBINED_RECOVERY_UNCONSTRAINED ?= 1
RECOVERY_INFERENCE_STEPS ?= 128
RECOVERY_ITERATIONS ?= 128
RECOVERY_FRAME_STRIDE ?= 1
RECOVERY_START_MODE ?= damage
RECOVERY_SEED_CELLS ?= 64
RECOVERY_CONFIDENCE_WINDOW ?= 24
RECOVERY_CONFIDENCE_REQUIRED ?= 20
RECOVERY_NO_PROGRESS_PATIENCE ?= 32
RECOVERY_EXTRA_STEPS_AFTER_COMPLETE ?= 32
RECOVERY_CONSENSUS_MIN_VOTES ?= 1
RECOVERY_SINGLE_VOTE_CONFIDENCE ?= 0.99
RECOVERY_DIRECTION_PROBABILITY_THRESHOLD ?= 0.6
RECOVERY_UNCONSTRAINED ?= 1
RECOVERY_FORCE_DOWNLOAD ?= 1
DAMAGE_INFERENCE_REPO ?= shyamsn97/cube-regen-damage-detection
DAMAGE_INFERENCE_OUTPUT ?= examples/damage_predictions
DAMAGE_INFERENCE_STEPS ?= 128
SAKANA_TRAIN_SCRIPT ?= examples/sakana/train_sakana_damage.py
SAKANA_INFERENCE_SCRIPT ?= examples/sakana/infer_sakana_damage.py
SAKANA_REPO ?= shyamsn97/sakana-cube-regen-damage-detection
SAKANA_INFERENCE_OUTPUT ?= examples/sakana/outputs/inference
SAKANA_INFERENCE_STEPS ?= 128
SAKANA_DAMAGE_RECOVERY_OUTPUT ?= sakana_damage_recovery.gif
SAKANA_SEED_RECOVERY_OUTPUT ?= sakana_seed_recovery.gif
SAKANA_RECOVERY_DAMAGE_TYPE ?= sphere
SAKANA_RECOVERY_RADIUS ?= 3
SAKANA_RECOVERY_CENTER_FRACTIONS ?= 0.25 0.5 0.75
SAKANA_RECOVERY_ITERATIONS ?= 128
SAKANA_RECOVERY_FRAME_STRIDE ?= 4
SAKANA_RECOVERY_SEED_CELLS ?= 64
SAKANA_RECOVERY_CONFIDENCE_WINDOW ?= 24
SAKANA_RECOVERY_CONFIDENCE_REQUIRED ?= 24
SAKANA_RECOVERY_CONSENSUS_MIN_VOTES ?= 1
SAKANA_RECOVERY_SINGLE_VOTE_CONFIDENCE ?= 0.99
SAKANA_RECOVERY_UNCONSTRAINED ?= 1

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

visualize-shapenet-predictions: shapenet-subset-download ## render ShapeNet prediction rows from the combined HF model
	$(PYTHON) $(INFERENCE_SCRIPT) \
		--repo-id $(SHAPENET_PREDICTION_REPO) \
		--weights-filename $(SHAPENET_PREDICTION_WEIGHTS) \
		--config $(COMBINED_SHAPENET_MODAL_CONFIG) \
		--data-root $(SHAPENET_LOCAL_DIR) \
		--output-dir $(SHAPENET_PREDICTION_OUTPUT) \
		--num-samples $(SHAPENET_PREDICTION_SAMPLES)

visualize-damage-prediction: ## render pure damage-detection inference from the HF model
	$(PYTHON) $(INFERENCE_SCRIPT) \
		--repo-id $(DAMAGE_INFERENCE_REPO) \
		--config $(DAMAGE_CONFIG) \
		--output-dir $(DAMAGE_INFERENCE_OUTPUT) \
		--steps $(DAMAGE_INFERENCE_STEPS)

visualize-shape-conditioned-recovery-gif: ## render NPY sample recovery with shape-conditioned model
	$(PYTHON) $(SHAPE_CONDITIONED_RECOVERY_SCRIPT) \
		--repo-id $(SHAPE_CONDITIONED_RECOVERY_REPO) \
		--config $(SHAPE_CONDITIONED_RECOVERY_CONFIG) \
		--output-dir $(SHAPE_CONDITIONED_RECOVERY_OUTPUT_DIR) \
		--output $(SHAPE_CONDITIONED_RECOVERY_OUTPUT) \
		--use-adaptive-pooling $(SHAPE_CONDITIONED_USE_ADAPTIVE_POOLING) \
		--class-name "$(SHAPE_CONDITIONED_RECOVERY_CLASS_NAME)" \
		--class-label "$(SHAPE_CONDITIONED_RECOVERY_CLASS_LABEL)" \
		--shape-seed "$(SHAPE_CONDITIONED_RECOVERY_SHAPE_SEED)" \
		--sample-index $(SHAPE_CONDITIONED_RECOVERY_SAMPLE_INDEX) \
		--damage-types $(SHAPE_CONDITIONED_RECOVERY_DAMAGE_TYPES) \
		--damage-radius-range $(SHAPE_CONDITIONED_RECOVERY_DAMAGE_RADIUS_RANGE) \
		--num-damage-sites-range $(SHAPE_CONDITIONED_RECOVERY_NUM_DAMAGE_SITES_RANGE) \
		--min-damage-target-cells $(SHAPE_CONDITIONED_RECOVERY_MIN_DAMAGE_TARGET_CELLS) \
		--max-damage-attempts $(SHAPE_CONDITIONED_RECOVERY_MAX_DAMAGE_ATTEMPTS) \
		--steps $(RECOVERY_INFERENCE_STEPS) \
		--recovery-iterations $(RECOVERY_ITERATIONS) \
		--recovery-frame-stride $(RECOVERY_FRAME_STRIDE) \
		--recovery-start-mode $(RECOVERY_START_MODE) \
		--recovery-seed-cells $(RECOVERY_SEED_CELLS) \
		--recovery-confidence-window $(RECOVERY_CONFIDENCE_WINDOW) \
		--recovery-confidence-required $(RECOVERY_CONFIDENCE_REQUIRED) \
		--recovery-no-progress-patience $(RECOVERY_NO_PROGRESS_PATIENCE) \
		--recovery-extra-steps-after-complete $(RECOVERY_EXTRA_STEPS_AFTER_COMPLETE) \
		--recovery-consensus-min-votes $(RECOVERY_CONSENSUS_MIN_VOTES) \
		--recovery-single-vote-confidence $(RECOVERY_SINGLE_VOTE_CONFIDENCE) \
		--recovery-direction-probability-threshold $(RECOVERY_DIRECTION_PROBABILITY_THRESHOLD) $(if $(filter 0 false no,$(RECOVERY_UNCONSTRAINED)),--constrained-recovery,) \
		$(if $(filter 1 true yes,$(RECOVERY_FORCE_DOWNLOAD)),--force-download,) \
		$(if $(SHAPE_CONDITIONED_RECOVERY_DEVICE),--device $(SHAPE_CONDITIONED_RECOVERY_DEVICE),)

visualize-shape-conditioned-seed-recovery-gif: ## render shape-conditioned recovery from seed cells
	$(MAKE) visualize-shape-conditioned-recovery-gif RECOVERY_START_MODE=seed SHAPE_CONDITIONED_RECOVERY_OUTPUT=$(SHAPE_CONDITIONED_SEED_RECOVERY_OUTPUT)

visualize-shapenet-recovery-gif: shapenet-subset-download ## render ShapeNet sample recovery GIF
	$(PYTHON) $(SHAPENET_RECOVERY_SCRIPT) \
		--repo-id $(SHAPENET_RECOVERY_REPO) \
		--weights-filename $(SHAPENET_RECOVERY_WEIGHTS) \
		--config $(SHAPENET_RECOVERY_CONFIG) \
		--data-root $(SHAPENET_LOCAL_DIR) \
		--output-dir $(SHAPENET_RECOVERY_OUTPUT_DIR) \
		--output $(SHAPENET_RECOVERY_OUTPUT) \
		--category "$(SHAPENET_RECOVERY_CATEGORY)" \
		--sample-index $(SHAPENET_RECOVERY_SAMPLE_INDEX) \
		--steps $(RECOVERY_INFERENCE_STEPS) \
		--recovery-iterations $(RECOVERY_ITERATIONS) \
		--recovery-frame-stride $(RECOVERY_FRAME_STRIDE) \
		--recovery-start-mode $(RECOVERY_START_MODE) \
		--recovery-seed-cells $(RECOVERY_SEED_CELLS) \
		--recovery-confidence-window $(RECOVERY_CONFIDENCE_WINDOW) \
		--recovery-confidence-required $(RECOVERY_CONFIDENCE_REQUIRED) \
		--recovery-no-progress-patience $(RECOVERY_NO_PROGRESS_PATIENCE) \
		--recovery-extra-steps-after-complete $(RECOVERY_EXTRA_STEPS_AFTER_COMPLETE) \
		--recovery-consensus-min-votes $(RECOVERY_CONSENSUS_MIN_VOTES) \
		--recovery-single-vote-confidence $(RECOVERY_SINGLE_VOTE_CONFIDENCE) \
		--recovery-direction-probability-threshold $(RECOVERY_DIRECTION_PROBABILITY_THRESHOLD) $(if $(filter 0 false no,$(RECOVERY_UNCONSTRAINED)),--constrained-recovery,) \
		$(if $(filter 1 true yes,$(RECOVERY_FORCE_DOWNLOAD)),--force-download,)

visualize-shapenet-seed-recovery-gif: ## render ShapeNet recovery from seed cells
	$(MAKE) visualize-shapenet-recovery-gif RECOVERY_START_MODE=seed SHAPENET_RECOVERY_OUTPUT=$(SHAPENET_SEED_RECOVERY_OUTPUT)

visualize-combined-recovery-gif: ## render NPY sample recovery with combined model
	$(PYTHON) $(COMBINED_RECOVERY_SCRIPT) \
		--repo-id $(COMBINED_RECOVERY_REPO) \
		--config $(COMBINED_RECOVERY_CONFIG) \
		--output-dir $(COMBINED_RECOVERY_OUTPUT_DIR) \
		--output $(COMBINED_RECOVERY_OUTPUT) \
		--class-name "$(COMBINED_RECOVERY_CLASS_NAME)" \
		--class-label "$(COMBINED_RECOVERY_CLASS_LABEL)" \
		--shape-seed "$(COMBINED_RECOVERY_SHAPE_SEED)" \
		--sample-index $(COMBINED_RECOVERY_SAMPLE_INDEX) \
		--damage-types $(COMBINED_RECOVERY_DAMAGE_TYPES) \
		--damage-radius-range $(COMBINED_RECOVERY_DAMAGE_RADIUS_RANGE) \
		--num-damage-sites-range $(COMBINED_RECOVERY_NUM_DAMAGE_SITES_RANGE) \
		--min-damage-target-cells $(COMBINED_RECOVERY_MIN_DAMAGE_TARGET_CELLS) \
		--max-damage-attempts $(COMBINED_RECOVERY_MAX_DAMAGE_ATTEMPTS) \
		--steps $(RECOVERY_INFERENCE_STEPS) \
		--recovery-iterations $(RECOVERY_ITERATIONS) \
		--recovery-frame-stride $(RECOVERY_FRAME_STRIDE) \
		--recovery-start-mode $(RECOVERY_START_MODE) \
		--recovery-seed-cells $(RECOVERY_SEED_CELLS) \
		--recovery-confidence-window $(RECOVERY_CONFIDENCE_WINDOW) \
		--recovery-confidence-required $(RECOVERY_CONFIDENCE_REQUIRED) \
		--recovery-no-progress-patience $(RECOVERY_NO_PROGRESS_PATIENCE) \
		--recovery-extra-steps-after-complete $(RECOVERY_EXTRA_STEPS_AFTER_COMPLETE) \
		--recovery-consensus-min-votes $(RECOVERY_CONSENSUS_MIN_VOTES) \
		--recovery-single-vote-confidence $(RECOVERY_SINGLE_VOTE_CONFIDENCE) \
		--recovery-direction-probability-threshold $(RECOVERY_DIRECTION_PROBABILITY_THRESHOLD) $(if $(filter 0 false no,$(COMBINED_RECOVERY_UNCONSTRAINED)),--constrained-recovery,) \
		$(if $(filter 1 true yes,$(RECOVERY_FORCE_DOWNLOAD)),--force-download,)

visualize-combined-seed-recovery-gif: ## render combined model recovery from seed cells
	$(MAKE) visualize-combined-recovery-gif RECOVERY_START_MODE=seed COMBINED_RECOVERY_OUTPUT=$(COMBINED_SEED_RECOVERY_OUTPUT)

visualize-sakana-damage: ## render Sakana sphere/cube damage inference rows
	$(PYTHON) $(SAKANA_INFERENCE_SCRIPT) \
		--repo-id $(SAKANA_REPO) \
		--output-dir $(SAKANA_INFERENCE_OUTPUT) \
		--steps $(SAKANA_INFERENCE_STEPS)

visualize-sakana-recovery-gif: visualize-sakana-seed-recovery-gif ## render Sakana recovery from seed cells

visualize-sakana-damage-recovery-gif: ## render Sakana recovery from damaged full shape
	$(PYTHON) $(SAKANA_INFERENCE_SCRIPT) \
		--repo-id $(SAKANA_REPO) \
		--output-dir $(SAKANA_INFERENCE_OUTPUT) \
		--steps $(SAKANA_INFERENCE_STEPS) \
		--recovery-gif \
		--recovery-output $(SAKANA_DAMAGE_RECOVERY_OUTPUT) \
		--recovery-start-mode damage \
		--recovery-seed-cells $(SAKANA_RECOVERY_SEED_CELLS) \
		--recovery-damage-type $(SAKANA_RECOVERY_DAMAGE_TYPE) \
		--recovery-radius $(SAKANA_RECOVERY_RADIUS) \
		--recovery-center-fractions $(SAKANA_RECOVERY_CENTER_FRACTIONS) \
		--recovery-iterations $(SAKANA_RECOVERY_ITERATIONS) \
		--recovery-frame-stride $(SAKANA_RECOVERY_FRAME_STRIDE) \
		--recovery-confidence-window $(SAKANA_RECOVERY_CONFIDENCE_WINDOW) \
		--recovery-confidence-required $(SAKANA_RECOVERY_CONFIDENCE_REQUIRED) \
		--recovery-consensus-min-votes $(SAKANA_RECOVERY_CONSENSUS_MIN_VOTES) \
		--recovery-single-vote-confidence $(SAKANA_RECOVERY_SINGLE_VOTE_CONFIDENCE) $(if $(filter 0 false no,$(SAKANA_RECOVERY_UNCONSTRAINED)),--constrained-recovery,)

visualize-sakana-seed-recovery-gif: ## render Sakana recovery from starting cells
	$(PYTHON) $(SAKANA_INFERENCE_SCRIPT) \
		--repo-id $(SAKANA_REPO) \
		--output-dir $(SAKANA_INFERENCE_OUTPUT) \
		--steps $(SAKANA_INFERENCE_STEPS) \
		--recovery-gif \
		--recovery-output $(SAKANA_SEED_RECOVERY_OUTPUT) \
		--recovery-start-mode seed \
		--recovery-seed-cells $(SAKANA_RECOVERY_SEED_CELLS) \
		--recovery-damage-type $(SAKANA_RECOVERY_DAMAGE_TYPE) \
		--recovery-radius $(SAKANA_RECOVERY_RADIUS) \
		--recovery-center-fractions $(SAKANA_RECOVERY_CENTER_FRACTIONS) \
		--recovery-iterations $(SAKANA_RECOVERY_ITERATIONS) \
		--recovery-frame-stride $(SAKANA_RECOVERY_FRAME_STRIDE) \
		--recovery-confidence-window $(SAKANA_RECOVERY_CONFIDENCE_WINDOW) \
		--recovery-confidence-required $(SAKANA_RECOVERY_CONFIDENCE_REQUIRED) \
		--recovery-consensus-min-votes $(SAKANA_RECOVERY_CONSENSUS_MIN_VOTES) \
		--recovery-single-vote-confidence $(SAKANA_RECOVERY_SINGLE_VOTE_CONFIDENCE) $(if $(filter 0 false no,$(SAKANA_RECOVERY_UNCONSTRAINED)),--constrained-recovery,)

train: ## train with CONFIG=configs/train_combined.yaml (override CONFIG=...)
	$(PYTHON) $(TRAIN_SCRIPT) --config $(CONFIG)

train-modal: ## train with MODAL_CONFIG=configs/train_combined_modal.yaml
	$(PYTHON) $(TRAIN_SCRIPT) --config $(MODAL_CONFIG)

train-damage: ## train damage direction detection locally
	$(PYTHON) $(TRAIN_SCRIPT) --config $(DAMAGE_CONFIG)

train-damage-modal: ## train damage direction detection on Modal
	$(PYTHON) $(TRAIN_SCRIPT) --config $(DAMAGE_MODAL_CONFIG)

train-damage-shape-conditioned: ## train shape-conditioned damage detection locally
	$(PYTHON) $(TRAIN_SCRIPT) --config $(DAMAGE_SHAPE_CONDITIONED_CONFIG) --use-adaptive-pooling $(SHAPE_CONDITIONED_USE_ADAPTIVE_POOLING)

train-damage-shape-conditioned-modal: ## train shape-conditioned damage detection on Modal
	$(PYTHON) $(TRAIN_SCRIPT) --config $(DAMAGE_SHAPE_CONDITIONED_MODAL_CONFIG) --use-adaptive-pooling $(SHAPE_CONDITIONED_USE_ADAPTIVE_POOLING)

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

train-sakana: ## train the Sakana damage example locally
	$(PYTHON) $(SAKANA_TRAIN_SCRIPT)

train-sakana-modal: ## train the Sakana damage example on Modal
	$(PYTHON) $(SAKANA_TRAIN_SCRIPT) --mode modal --repo-id $(SAKANA_REPO)
