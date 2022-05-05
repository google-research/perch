# Chirp

A bioacoustics research project.

## Installation

You might need the following dependencies.

```bash
# Install Poetry for package management
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies for librosa (required for testing only)
sudo apt-get install libsndfile1
```

Afterwards you should be able to run `poetry install` to create a virtual
environment in which you can run the Chirp codebase. To run the tests, try
`poetry run pytest chirp/tests`.

*This is not an officially supported Google product.*
