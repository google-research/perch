# Perch

![CI](https://github.com/google-research/perch/actions/workflows/ci.yml/badge.svg)

A bioacoustics research project.

## Installation

You might need the following dependencies.

```bash
# Install Poetry for package management
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies for librosa
sudo apt-get install libsndfile1 ffmpeg

# Install all dependencies specified in the poetry configs.
poetry install
```

Running `poetry install` creates a virtual environment and installs all
dependencies, in which you can run the Perch codebase. To run the tests, try

```bash
poetry run python -m unittest discover -s chirp/tests -p "*test.py"
```

## BIRB data preparation

### Evaluation data

After [installing](#installation) the `chirp` package, run the following command from the repository's root directory:

```bash
poetry run tfds build -i chirp.data.bird_taxonomy,chirp.data.soundscapes \
    soundscapes/{ssw,hawaii,coffee_farms,sierras_kahl,high_sierras,peru}_full_length \
    bird_taxonomy/{downstream_full_length,class_representatives_slice_peaked}
```

The process should take 36 to 48 hours to complete and use around 256 GiB of disk space.


## Using a container

Alternatively, you can install and run this project using a container via
`docker` or `podman`. To build a container using the tag `perch`, run:
```bash
docker build . --tag perch
```

And then you can run the different commands using the container you just built.
For example, to run the tests, try:
```bash
docker run --rm -t perch python -m unittest discover -s chirp/tests -p "*test.py"
```

*This is not an officially supported Google product.*
