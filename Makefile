IMAGE ?= psyb0t/imagies
CUDA_IMAGE ?= $(IMAGE):latest-cuda
GENERATION_IMAGE ?= $(IMAGE):latest-generation
GENERATION_CUDA_IMAGE ?= $(IMAGE):latest-generation-cuda
CUDA_ARCHITECTURES ?= all-major
DOCS_IMAGE := node:24.21.0-bookworm-slim@sha256:d6aa754f16b3197301076f047b5def2f02ea1dbbc2ca920407d46d7ec7f87b20
DOCS_RUN := docker run --rm --cap-drop ALL --security-opt no-new-privileges \
	--user "$$(id -u):$$(id -g)" -e HOME=/tmp -e COREPACK_HOME=/work/website/.corepack \
	-v "$(CURDIR):/work:ro" -v "$(CURDIR)/website:/work/website" -w /work/website
OPENAPI_FILE ?= docs/reference/openapi.json
DEV_IMAGE := imagies-dev:local
GO_IMAGE := golang:1.26-bookworm@sha256:a688600ca24f8a4d3ca77f95b0dd40704a9fc787c826660eb7ba0b641b8b175d
DEV_RUN := docker run --rm --user $(shell id -u):$(shell id -g) \
	-e HOME=/tmp -v "$(CURDIR):/work" -w /work $(DEV_IMAGE)
BUILDFLAGS := $(if $(filter 1,$(EDITING_ONLY)),--build-arg IMAGIES_VARIANT=editing,)

.PHONY: help build build-cuda build-generation build-generation-cuda dev-image format lint test test-worker check

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build the runtime image
	docker build $(BUILDFLAGS) -t $(IMAGE) .

build-cuda: ## Build the NVIDIA vision image
	docker build --build-arg ACCELERATOR=cuda -t $(CUDA_IMAGE) .

build-generation: ## Build Imagies with CPU image generation
	docker build -f Dockerfile.generation $(BUILDFLAGS) -t $(GENERATION_IMAGE) .

build-generation-cuda: ## Build Imagies with NVIDIA image generation
	docker build -f Dockerfile.generation $(BUILDFLAGS) \
		--build-arg ACCELERATOR=cuda --build-arg CUDA_ARCHITECTURES="$(CUDA_ARCHITECTURES)" \
		-t $(GENERATION_CUDA_IMAGE) .

dev-image: ## Build the development image
	docker build -f Dockerfile.dev -t $(DEV_IMAGE) .

format: dev-image ## Format Python source and tests
	$(DEV_RUN) ruff format src tests

lint: dev-image ## Check formatting and lint
	$(DEV_RUN) ruff check src tests
	$(DEV_RUN) ruff format --check src tests

test: dev-image ## Run API and MCP tests
	$(DEV_RUN) pytest -q -p no:cacheprovider

test-worker: ## Check the generation worker without network access
	docker run --rm --network=none --user $(shell id -u):$(shell id -g) \
		-e HOME=/tmp -e GOCACHE=/tmp/go-cache \
		-v "$(CURDIR)/generation/worker:/work:ro" -w /work $(GO_IMAGE) \
		sh -ec 'test -z "$$(gofmt -l cmd internal)"; go vet -mod=vendor ./...; go test -mod=vendor -race ./...'

check: lint test test-worker ## Run lint and tests

.PHONY: docs-build docs-openapi docs-check-openapi

docs-openapi: dev-image ## Export the OpenAPI document from FastAPI
	mkdir -p docs/reference
	docker run --rm --network none -e IMAGIES_GENERATION=1 -e IMAGIES_DATA_DIR=/tmp/imagies-docs -e PYTHONPATH=/app/src -v "$(CURDIR)/src:/app/src:ro" $(DEV_IMAGE) python -c 'import json; from imagies.server import app; print(json.dumps(app.openapi(), indent=2, sort_keys=True))' > "$(OPENAPI_FILE)"

docs-check-openapi: dev-image ## Check that the OpenAPI document matches the routes
	docker run --rm --network none -e IMAGIES_GENERATION=1 -e IMAGIES_DATA_DIR=/tmp/imagies-docs -e PYTHONPATH=/app/src -v "$(CURDIR)/src:/app/src:ro" $(DEV_IMAGE) python -c 'import json; from imagies.server import app; print(json.dumps(app.openapi(), indent=2, sort_keys=True))' | diff -u "$(OPENAPI_FILE)" -

docs-build: docs-check-openapi ## Build and check the documentation site
	$(DOCS_RUN) $(DOCS_IMAGE) corepack pnpm install --frozen-lockfile --ignore-scripts --store-dir .pnpm-store
	$(DOCS_RUN) --network none -e OPENAPI_FILE="$(OPENAPI_FILE)" $(DOCS_IMAGE) corepack pnpm run build
