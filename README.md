# Perch

![CI](https://github.com/google-research/perch/actions/workflows/ci.yml/badge.svg)

A bioacoustics research project.

## Directory of Things

We have published quite a few things which utilize this repository!

### Perch (and SurfPerch!)

We produce a bird species classifier, trained on over 10k species.

* The current [released Perch model](https://www.kaggle.com/models/google/bird-vocalization-classifier/frameworks/tensorFlow2/variations/bird-vocalization-classifier) is available from Kaggle Models.
* The current-best citation for the model is our paper: [Global birdsong embeddings enable superior transfer learning for bioacoustic classification](https://www.nature.com/articles/s41598-023-49989-z.epdf).
* The [SurfPerch model](https://www.kaggle.com/models/google/surfperch), trained on a combination of birds, coral reef sounds, and general audio, is also available at Kaggle models. The associated paper is (as of this writing) [available as a preprint](https://arxiv.org/abs/2404.16436).

The major parts of the Perch model training code is broken up across the following files:

* [Model frontend](https://github.com/google-research/perch/blob/main/chirp/models/frontend.py) - we use a PCEN melspectrogram.
* [EfficientNet model](https://github.com/google-research/perch/blob/main/chirp/models/efficientnet.py)
* [Training loop](https://github.com/google-research/perch/blob/main/chirp/train/classifier.py)
* [Training launch script](https://github.com/google-research/perch/blob/main/chirp/projects/main.py)
* [Export](https://github.com/google-research/perch/blob/main/chirp/export_utils.py) from JAX to Tensorflow and TFLite

### Agile Modeling

Agile modeling combines search and active learning to produce classifiers for novel concepts quickly.

Here's [Tutorial Colab Notebook](https://colab.research.google.com/drive/1gPBu2fyw6aoT-zxXFk15I2GObfMRNHUq) we produced for [Climate Change AI](https://www.climatechange.ai/) and presented at their workshop at [NeurIPS 2023](https://www.climatechange.ai/papers/neurips2023/133).

We maintain three 'working' notebooks for agile modeling in this repository:

* [`embed_audio.ipynb`](https://github.com/google-research/perch/blob/main/embed_audio.ipynb) for performing mass-embedding of audio.
* [`agile_modeling.ipynb`](https://github.com/google-research/perch/blob/main/agile_modeling.ipynb) for search and active learning over embeddings.
* [`analysis.ipynb`](https://github.com/google-research/perch/blob/main/analysis.ipynb) for running inference and performing call density estimation (see below).
* The code for agile modeling is largely contained in the [inference directory](https://github.com/google-research/perch/tree/main/chirp/inference), which contains its own extensive README.

The agile modeling work supports a number of different models, including our models (Perch and SurfPerch, and the [multi-species whale classifier](https://www.kaggle.com/models/google/multispecies-whale)), BirdNet, and some general audio models like [YamNet](https://www.kaggle.com/models/google/yamnet) and [VGGish](https://www.kaggle.com/models/google/vggish). Adding support for additional models is fairly trivial.

### Call Density

We provide some tooling for estimating the proportion of audio windows in a dataset containing a target call type or species - anything you have a classifier for.

Paper: [All Thresholds Barred: Direct Estimation of Call Density in Bioacoustic Data](https://www.frontiersin.org/journals/bird-science/articles/10.3389/fbirs.2024.1380636/full)

Code: See [call_density.py](https://github.com/google-research/perch/blob/main/chirp/inference/call_density.py) and [call_density_test.py](https://github.com/google-research/perch/blob/main/chirp/inference/tests/call_density_test.py). Note that the code contains some interactions with our broader 'agile modeling' work, though we have endeavoured to isolate the underlying mathematics in more modular functions.

### BIRB Benchmark

We produced a benchmark paper for understanding model generalization when transferring from focal to passive acoustic datasets. The preprint is [available here](https://arxiv.org/abs/2312.07439).

For details on setting up the benchmark and evaluation protocol, please refer to this [brief readme](https://docs.google.com/document/d/1RasVkxIKKlUToFlJ8gZxaHqIE-mMy9G1MZwfK98Gb-I) with instructions. The evaluation codebase is in [perch/chirp/eval](https://github.com/google-research/perch/tree/main/chirp/eval).

To build the BIRB evaluation data, after [installing](#installation) the `chirp` package, run the following command from the repository's root directory:

```bash
poetry run tfds build -i chirp.data.bird_taxonomy,chirp.data.soundscapes \
    soundscapes/{ssw,hawaii,coffee_farms,sierras_kahl,high_sierras,peru}_full_length \
    bird_taxonomy/{downstream_full_length,class_representatives_slice_peaked}
```

The process should take 36 to 48 hours to complete and use around 256 GiB of disk space.


### Source-Free Domain Adaptation and NOTELA

We have a paper on 'source-free domain generalization,' which involves automatic model adaptation to data from a shifted domain. We have a [blog post](https://research.google/blog/in-search-of-a-generalizable-method-for-source-free-domain-adaptation/) where you can read more about it. The [paper](https://proceedings.mlr.press/v202/boudiaf23a.html) was published in ICML 2023. The code for this project has been archived. You can [download a snapshot](https://github.com/google-research/perch/releases/tag/sfda-codebase-snapshot) of the repository containing the code, which can be found in the `chirp/projects/sfda` directory.


## Installation

We support installation on a generic Linux workstation.
A GPU is recommended, especially when working with large datasets.
The recipe below is the same used by our continuous integration testing.

Some users have successfully used our repository with the Windows Linux
Subsystem, or with Docker in a cloud-based virtual machine. Anecdotally,
installation on OS X is difficult.

You will need the following dependencies.

```bash
# Install Poetry for package management
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies for librosa
sudo apt-get install libsndfile1 ffmpeg

# Install all dependencies specified in the poetry configs.
# Note that for Windows machines, you can remove the `--with nonwindows`
# option to drop some optional dependencies which do not build for Windows.
poetry install  --with jaxtrain --with nonwindows
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

*This is not an officially supported Google product.*
