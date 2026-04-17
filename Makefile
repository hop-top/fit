.PHONY: test test-e2e-smoke test-e2e-full test-e2e-gpu lint

test:
	cd py && python -m pytest tests/ -m "not slow and not gpu" -q

lint:
	cd py && python -m ruff check src/ tests/

test-e2e-smoke:
	cd py && python -m pytest tests/e2e/ -m "not slow and not gpu" -q

test-e2e-full:
	cd py && python -m pytest tests/e2e/ -q

test-e2e-gpu:
	cd py && python -m pytest tests/e2e/ -m gpu -q
