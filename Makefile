SHELL := /bin/bash

PROJECT_ROOT := $(CURDIR)
CLI := uv run python -m src.cli
PYTHON := uv run python

OVERWRITE ?= 0
WORKERS ?= 32
MATCH_WORKERS ?= 32
ifeq ($(OVERWRITE),1)
SCAN_PEOPLE_ARGS := --overwrite
else
SCAN_PEOPLE_ARGS :=
endif

SAMPLE_PAGES_DIR := $(PROJECT_ROOT)/data/pages_sample
SAMPLE_OBSIDIAN_DIR := $(PROJECT_ROOT)/data/obsidian_sample
SAMPLE_CACHE_DIR := $(PROJECT_ROOT)/data/cache_sample
SAMPLE_IMAGES_SRC := $(PROJECT_ROOT)/data/obsidian/images
SAMPLE_IMAGES_DEST := $(SAMPLE_OBSIDIAN_DIR)/images

SAMPLE_ENV := PAGES_DIR=$(SAMPLE_PAGES_DIR) \
		OBSIDIAN_PERSONS_DIR=$(SAMPLE_OBSIDIAN_DIR)/persons \
		OBSIDIAN_ITEMS_DIR=$(SAMPLE_OBSIDIAN_DIR)/items \
		CACHE_DIR=$(SAMPLE_CACHE_DIR) \
		SGR_LOGGING=DEBUG

.PHONY: run_full run_sample validate_sample validate_sample_verbose

run_full:
	$(CLI) ner-extract
	$(CLI) scan-people $(SCAN_PEOPLE_ARGS) --workers $(WORKERS)
	$(CLI) cluster --match-workers $(MATCH_WORKERS)
	$(CLI) gen-person-notes
	$(CLI) link-persons

run_sample:
	rm -rf $(SAMPLE_CACHE_DIR) $(SAMPLE_OBSIDIAN_DIR)
	mkdir -p $(SAMPLE_OBSIDIAN_DIR)/persons $(SAMPLE_OBSIDIAN_DIR)/items $(SAMPLE_IMAGES_DEST)
	$(SAMPLE_ENV) $(CLI) ner-extract
	$(SAMPLE_ENV) $(CLI) scan-people --overwrite --workers $(WORKERS)
	$(SAMPLE_ENV) $(CLI) cluster --match-workers $(MATCH_WORKERS)
	$(SAMPLE_ENV) $(CLI) gen-person-notes
	$(SAMPLE_ENV) $(CLI) link-persons
	$(PYTHON) scripts/copy_sample_images.py \
		--pages-dir $(SAMPLE_PAGES_DIR) \
		--source-images-dir $(SAMPLE_IMAGES_SRC) \
		--dest-images-dir $(SAMPLE_IMAGES_DEST)

validate_sample:
	$(PYTHON) scripts/validate_pipeline.py --samples-dir tests/samples/pages

validate_sample_verbose:
	$(PYTHON) scripts/validate_pipeline.py --samples-dir tests/samples/pages --verbose
