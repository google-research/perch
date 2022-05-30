# Chirp

![CI](https://github.com/google-research/chirp/actions/workflows/ci.yml/badge.svg)

A bioacoustics research project.

## Installation

You might need the following dependencies.

```bash
# Install Poetry for package management
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies for librosa (required for testing only)
sudo apt-get install libsndfile1
```

Afterwards you should be able to run `poetry update tf-nightly && poetry
install` to create a virtual environment in which you can run the Chirp
codebase. To run the tests, try

```bash
poetry run python -m unittest discover -s chirp/tests -p "*test.py"
```

*This is not an officially supported Google product.*
