.PHONY: install
install:
	uv sync --all-packages --all-extras

.PHONY: update
update:
	uv sync --all-packages --all-extras -U

.PHONY: lint
lint:
	uv run ruff check . && uv run ty check .

.PHONY: test
test:
	uv run pytest
