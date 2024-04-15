# Perch

![CI](https://github.com/google-research/perch/actions/workflows/ci.yml/badge.svg)

A bioacoustics research project.

## Installation

We support installation on a generic Linux workstation.
A GPU is recommended, especially when working with large datasets.
The recipe below is the same used by our continuous integration testing.

Some users have successfully used our repository with the Windows Linux
Subsystem, or with Docker in a cloud-based virtual machine. Anecdotally,
installation on OS X is difficult.

You might need the following dependencies.

```bash
# Install Poetry for package management
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies for librosa
sudo apt-get install libsndfile1 ffmpeg

# Install all dependencies specified in the poetry configs
poetry install  --with jaxtrain
```

Running `poetry install` installs all Perch dependencies into a new virtual environment, in which you can run the Perch code base. To run the tests, use:

```bash
poetry run python -m unittest discover -s chirp/tests -p "*test.py"
poetry run python -m unittest discover -s chirp/inference/tests -p "*test.py"
```

### Lightweight Inference

Note that if you only need the python notebooks for use with pre-trained models,
you can install with lighter dependencies:

```
# Install inference-only dependencies specified in the poetry configs
poetry install
```

And check that the inference tests succeed:
```bash
poetry run python -m unittest discover -s chirp/inference/tests -p "*test.py"
```

## Using a container

Alternatively, you can install and run this project using a container via Docker. To build a container using the tag `perch`, run:

```bash
git clone https://github.com/google-research/perch
cd perch
docker build . --tag perch
```

After building the container, to run the unit tests, use:

```bash
docker run --rm -t perch python -m unittest discover -s chirp/tests -p "*test.py"
```

## BIRB benchmark

### Data preparation
To build the BIRB evaluation data, after [installing](#installation) the `chirp` package, run the following command from the repository's root directory:

```bash
poetry run tfds build -i chirp.data.bird_taxonomy,chirp.data.soundscapes \
    soundscapes/{ssw,hawaii,coffee_farms,sierras_kahl,high_sierras,peru}_full_length \
    bird_taxonomy/{downstream_full_length,class_representatives_slice_peaked}
```

The process should take 36 to 48 hours to complete and use around 256 GiB of disk space.

### Benchmark README
For details on setting up the benchmark and evaluation protocol, please refer to this [brief readme](https://docs.google.com/document/d/1RasVkxIKKlUToFlJ8gZxaHqIE-mMy9G1MZwfK98Gb-I) with instructions. The evaluation codebase is in [perch/chirp/eval](https://github.com/google-research/perch/tree/main/chirp/eval).

*This is not an officially supported Google product.*
