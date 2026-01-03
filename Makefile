.PHONY: install install-dev test test-cov lint format clean build publish bump-major bump-minor bump-patch release

# Install the package
install:
	pip install -e .

# Install with dev/test dependencies
install-dev:
	pip install -e ".[test,async]"

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=inferencesh --cov-report=term-missing

# Run a specific test file
test-file:
	@test -n "$(FILE)" || (echo "Usage: make test-file FILE=tests/test_client.py" && exit 1)
	pytest $(FILE) -v

# Lint the code
lint:
	python -m flake8 src/inferencesh tests --max-line-length=100

# Type check
typecheck:
	python -m mypy src/inferencesh --ignore-missing-imports

# Format code (requires black)
format:
	python -m black src/inferencesh tests examples

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Build package
build: clean
	python -m build

# Publish to PyPI (requires twine)
publish: build
	python -m twine upload dist/*

# Run example
example:
	cd examples && python agent_chat.py

# Quick sanity check - import the package
check:
	python -c "from inferencesh import inference, AgentRuntimeConfig, is_terminal_status; print('✓ All imports OK')"

# Version bumping (commits, tags, and pushes)
bump-major:
	./scripts/bump.sh major

bump-minor:
	./scripts/bump.sh minor

bump-patch:
	./scripts/bump.sh patch

# Create GitHub release (requires gh CLI and being on main branch)
release:
	./scripts/release.sh

