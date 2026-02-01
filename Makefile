.PHONY: install
install:
	uv sync --all-packages --all-extras

.PHONY: update
update:
	uv sync --all-packages --all-extras -U

.PHONY: test
test:
	uv run pytest
