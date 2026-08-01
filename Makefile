.PHONY: install lint test quick-check train sweep

install:
	uv sync --extra test

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

quick-check:
	uv run python -m memory_tokens.train --steps 40 --quick-check --output artifacts/model.pt

train:
	uv run python -m memory_tokens.train --steps 1000 --output artifacts/model.pt

sweep:
	uv run python -m memory_tokens.experiment sweep --checkpoint artifacts/model.pt --output artifacts/sweep.json
