# BIRB

The BIRB benchmark.

## Installation

You might need the following dependencies.

```bash
# Install Poetry for package management
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies for librosa (required for testing only)
sudo apt-get install libsndfile1

# Install all dependencies specified in the poetry configs.
poetry install
```

Running `poetry install` creates a virtual environment and installs all
dependencies, in which you can run the `birb` codebase.
