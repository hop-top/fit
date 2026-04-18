GO_DIR   := go
TS_DIR   := ts
PY_DIR   := py
RS_DIR   := rs
PHP_DIR  := php

.PHONY: default check install build test lint \
        install\:go install\:ts install\:py install\:rs install\:php \
        build\:go build\:ts build\:rs \
        test\:go test\:ts test\:py test\:rs test\:php \
        test-e2e-smoke test-e2e-full test-e2e-gpu \
        test-workflow \
        lint\:go lint\:ts lint\:py lint\:rs lint\:php \
        format typecheck links setup parity

# --- Aggregates ---

default: install build check

check: lint typecheck test

# --- Install ---

install: install\:go install\:ts install\:py install\:rs install\:php

install\:go:
	cd $(GO_DIR) && go mod download

install\:ts:
	cd $(TS_DIR) && pnpm install --ignore-workspace --frozen-lockfile \
		|| cd $(TS_DIR) && pnpm install --ignore-workspace

install\:py:
	@python -c "import fit" 2>/dev/null \
		|| (cd $(PY_DIR) && python -m pip install -e ".[dev]" --quiet \
		    || cd $(PY_DIR) && pip install -e ".[dev]" --quiet)

install\:rs:
	cd $(RS_DIR) && cargo fetch

install\:php:
	cd $(PHP_DIR) && composer install --no-interaction --quiet

# --- Build ---

build: build\:ts build\:rs build\:go

build\:go:
	cd $(GO_DIR) && go build ./...

build\:ts:
	cd $(TS_DIR) && npx tsc

build\:rs:
	cd $(RS_DIR) && cargo build

# --- Test ---

test: test\:py test\:go test\:ts test\:rs test\:php

test\:py:
	cd $(PY_DIR) && PYTHONPATH=src python -m pytest tests/ -x -q

test\:go:
	cd $(GO_DIR) && go test ./... -count=1

test\:ts:
	cd $(TS_DIR) && npx vitest run

test\:rs:
	cd $(RS_DIR) && cargo test

test\:php:
	cd $(PHP_DIR) && vendor/bin/phpunit --no-coverage

# --- E2E ---

test-e2e-smoke:
	cd $(PY_DIR) && PYTHONPATH=src python -m pytest tests/e2e/ -m "not slow and not gpu" -q

test-e2e-full:
	cd $(PY_DIR) && PYTHONPATH=src python -m pytest tests/e2e/ -q

test-e2e-gpu:
	cd $(PY_DIR) && PYTHONPATH=src python -m pytest tests/e2e/ -m gpu -q

# --- Workflow shell tests ---

test-workflow:
	bats .github/tests/

# --- Lint ---

lint: lint\:py lint\:go lint\:ts lint\:rs lint\:php

lint\:py:
	cd $(PY_DIR) && ruff check src/ tests/

lint\:go:
	cd $(GO_DIR) && go vet ./...

lint\:ts:
	cd $(TS_DIR) && npx tsc --noEmit

lint\:rs:
	cd $(RS_DIR) && cargo clippy -- -D warnings

lint\:php:
	cd $(PHP_DIR) && vendor/bin/phpstan analyse src/

# --- Format ---

format:
	cd $(PY_DIR) && ruff format src/ tests/
	cd $(GO_DIR) && gofmt -w .
	cd $(TS_DIR) && npx prettier --write .
	cd $(RS_DIR) && cargo fmt
	cd $(PHP_DIR) && vendor/bin/php-cs-fixer fix src/

# --- Typecheck ---

typecheck:
	cd $(PY_DIR) && mypy src/

# --- Links ---

links:
	@if command -v lychee >/dev/null 2>&1; then \
		lychee --no-progress .; \
	else \
		echo "lychee not installed; skipping link check"; \
	fi

# --- Setup ---

setup: install
	@if [ -d .githooks ]; then git config core.hooksPath .githooks; fi

# --- Parity ---

parity:
	cd $(GO_DIR) && go test -tags parity ./cmd/fit/... -v
