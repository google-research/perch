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

# Install all dependencies specified in the poetry configs
poetry install
```

Running `poetry install` installs all Perch dependencies into a new virtual environment, in which you can run the Perch code base. To run the tests, use:

```bash
poetry run python -m unittest discover -s chirp/tests -p "*test.py"
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

## BIRB data preparation

To build the BIRB evaluation data, after [installing](#installation) the `chirp` package, run the following command from the repository's root directory:

```bash
poetry run tfds build -i chirp.data.bird_taxonomy,chirp.data.soundscapes \
    soundscapes/{ssw,hawaii,coffee_farms,sierras_kahl,high_sierras,peru}_full_length \
    bird_taxonomy/{downstream_full_length,class_representatives_slice_peaked}
```

The process should take 36 to 48 hours to complete and use around 256 GiB of disk space.

*This is not an officially supported Google product.*
