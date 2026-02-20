.PHONY: test package

# Run all tests
test:
	python -m pytest

# Package selected folders/files into a zip
package:
	mkdir -p dist
	zip -r dist/embedding-trainer.zip src/ configs/ scripts/ pyproject.toml README.md LICENSE
