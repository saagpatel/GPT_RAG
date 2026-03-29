.PHONY: install test lint clean run

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ tests/

run:
	python -m src

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; rm -rf .pytest_cache
