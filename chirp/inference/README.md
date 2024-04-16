# Chirp Inference

This library is for applying trained models to data.

## Notebooks for Transfer Learning

We provide a few Python notebooks for efficient transfer learning, as suggested
in [Feature Embeddings from Large-Scale Acoustic Bird Classifiers Enable Few-Shot Transfer Learning](https://arxiv.org/abs/2307.06292).

The full workflow is illustrated in a
[Colab tutorial](https://colab.research.google.com/drive/1gPBu2fyw6aoT-zxXFk15I2GObfMRNHUq).
This tutorial can be used with Google Colab's free-tier, requiring no software
installation, though a (free) Google account is required. This notebook can
be copied and adapted to work with your own data, stored in Drive.

For local installation and use of the base Python notebooks, we recommend using
a Linux machine (eg, Ubuntu) with a moderate GPU. Our continuous integration
tests install and run on Linux, so that is your best bet for compatibility.
Some users have had success using the Windows Linux Subsystem (WSL), or with
using Docker and virtual machines hosted in the cloud.
Anecdotally, installation on OS X is difficult.

### Workflow Overview

The classifier workflow has two-or-three steps. We work with an
/embedding model/, a large quantity of /unlabeled data/ and a usually-smaller
set of /labeled data/.

We first need to compute /embeddings/ of the target unlabeled audio. The
unlabeled audio is specified by one or more 'globs' of files like:
`/my_home/audio/*/*.flac`. Any audio formats readable by Librosa should be fine.
We provide `embed_audio.ipynb` to do so. This creates a dataset of embeddings
in a directory of your choice, along with a configuration file.
Computing embeddings can take a while for large datasets;
we suggest using a machine with a GPU. For truly massive datasets (terabytes
or more), we provide a Beam pipeline via `embed.py` which can run on a cluster.
Setting this up may be challenging, however; feel free to get in touch if you
have questions.

Once we have embedded the unlabeled audio, you can use `agile_modeling.ipynb`
to search for interesting audio and create a classifier. Starting from a clip
(or Xeno-Canto id, or URL for an audio file), you can search for similar audio
in the unlabeled data.
By providing a label and clicking on relevant results, you will start amassing
a set of `labeled data`.

You can also add labeled data manually. The labeled data is stored in a simple
'folder-of-folders' format: each class is given a folder, whose name is the
class. (Explicit negatives can be put in an 'unknown' folder.) This makes it
easy to add additional examples. It is recommended to add examples with length
matching the /window size/ of the embedding model (5 seconds for Perch, or
3 seconds for BirdNET).

From there, the notebook will build a small classifier using
the embeddings of the labeled audio. The classifier can then be run on the
unlabeled data. Hand-labeling results will allow you to feed new data into the
labeled dataset, and iterate quickly.

The `analysis.ipynb` notebook provides additional tools for analyzing data with
a pre-trained classifier, as developed in `agile_modeling.ipynb`. It can be
used to run detections over new data, estimate total call density, and
evaluate the real-world model quality.

### Installation (Linux)

Install the main repository, following the [instructions](https://github.com/google-research/perch) in the main README.md, and check that the tests pass.

Separately, install [Jupyter](https://jupyter.org/). Once Jupyter and Chirp are
installed, use the command line to navigate to the directory where Chirp is
installed, and launch the notebook with:
```
poetry run jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 --NotebookApp.port_retries=0
```
This starts the notebook server. A link to the notebook should appear in the
terminal output; open this in a web browser.

Once in the Jupyter interface, navigate to `chirp/inference`, and get started
with `embed_audio.ipynb`.

Note that you can use Google Colab to get some nicer notebook layout. Copy the
notebook file into Google Drive and open it with Colab. Then use the
`Connect to a Local Runtime` option to connect to your Jupyter notebook server.

## The Embedding Model Interface

We provide a model wrapping interface `interface.EmbeddingModel` which can be
implemented by a wide range of models providing some combination of
classification logits, embeddings, and separated audio. Implementations are
provided in `models.py`, including:
* a `PlaceholderModel` which can be used for testing,
* `TaxonomyModelTF`: an exported Chirp classifier SavedModel,
* `SeparatorModelTF`: an exported Chirp separation model,
* `BirdNet`: applies the BirdNet saved model, which can be obtained from the
  BirdNET-Analyzer git repository.
* `BirbSepModelTF1`: Applies the separation model described in [the Bird MixIT
  paper](https://arxiv.org/abs/2110.03209)
* `SeparateEmbedModel`: Combines different separation and embedding/inference
  models, by separating the target audio and then embedding each separate
  channel. If the embedding model produces logits, the max logits are taken
  over the separated channels.

The primary function in the `EmbeddingModel` interface is
`EmbeddingModel.embed(audio_array)` which runs model inference on the provided
audio array. The outputs are an `interface.InferenceOutputs` instance, which
contains optional embeddings, logits, and separated audio.

# Inference Pipeline

The `embed.py` script contains a Beam pipeline for running an `EmbeddingModel`
on a large collection of input audio. The pipeline produces a database of examples
in [TFRecord format](https://www.tensorflow.org/tutorials/load_data/tfrecord).
Example configurations can be found in the `chirp/configs/inference` directory.

Currently configuration is handled by supplying a config name on the command
line (one of `raw_soundscapes`, `separate_soundscapes` or
`birdnet_soundscapes`). The corresponding configuration file in
`chirp/configs/inference` can be edited to provide the model location and
glob matching pattern for the target wav files.

The embedding script also includes a `dry_run` option which processes a single
file at random using the chosen configuration. This is useful for ensuring that
the model and data is configured properly before launching a large job.

Step-by-step:

* Run the main repository's installation instructions.

* Download and extract the Perch model from TFHub:
  https://tfhub.dev/google/bird-vocalization-classifier/

* Adjust the inference raw_soundscapes config file:

    * Fill in `config.source_file_patterns` with the path to some audio
      files. eg: `config.source_file_patterns = ['/my/drive/*.wav']`

    * Fill in the `model_checkpoint_path` with the path of the model
      downloaded from TFHub.

    * Fill in `config.output_dir` with the path where you would like to
      write the outputs. eg, `config.output_dir = '/my/drive/embeddings'`

    * Adjust `config.shard_length_s` and `config.num_shards_per_file` according
      to your target data. We produce work-units for each audio file by breaking
      each file into parts according to these config values: Setting
      `shard_length_s` to 60 means each work unit will handle 60 seconds of
      audio from a given file. Setting `num_shards_per_file` to 15 will then
      produce a work-unit for each of the first 15 minutes of the audio.
      If the audio is less than 15 minutes, these extra work units will just do
      nothing. If the audio is more than 15 minutes long, the extra audio
      will not be used.

* From the terminal, change directory to the main chirp repository, and use
  poetry to run the embed.py script:
  ```poetry run python chirp/inference/embed.py --```

# Agile Modeling Code Structure

The agile modeling python notebooks heavily rely on the code in this directory.
The three parts of the workflow are embedding, search, and classification. The
latter two parts generally require knowing where the embeddings are, how
to join embeddings with their source audio, and how to display examples
to users in Colab.

Embedding is handled by utilities in `embed_lib.py`. When embeddings are
computed, a configuration file is written beside the embeddings which indicates
the embedding model used and the audio file globs which were embedded.

## Boostrap Config and Project State

For subsequent steps, we coordinate activity with `search/bootstrap.py`. First,
we create a `bootstrap.BootstrapConfig`, which collects info on the embedding
model, location of embeddings files, and audio glob (for connecting embeddings
with their source audio). The `bootstrap.BootstrapConfig` is then used to create
a `bootstrap.BootstrapState`, which includes an instantiated copy of the
embedding model, and is also used to create certain objects which depend heavily
on the configuration - such as the Tensorflow Dataset of embeddings, or an
iterator over audio files corresponding to embeddings.

## Search

Brute-force search is handled by `bootstrap/search.py`. This is optimized for
fast execution, and is fairly adaptable to new situations, such as searching
with a classifier, finding examples at a specific distance from the query,
or selecting random examples.

The `search.TopKSearchResults` object maintains a list of `search.SearchResult`,
both of which are also important for display.

## Classification

Users may provide data in a 'folder-of-folders' format. The
`classify.data_lib.py` file contains utilities for loading embeddings of
labeled data into memory in a `MergedDataset` object. This object contains
everything needed for training small classifiers on top of embeddings.
Actual small-model training code is contained in `classify/classify.py`.

## Display

Displaying examples to the user generally requires connecting embeddings with
source audio. The `bootstrap.BootstrapState` is responsible for providing an
iterator (via `search_results_audio_iterator`) which iterates over and
`search.TopKSearchResults` object, attaching audio to each result. The iterator
provides results in the same order as they appear in the
`search.TopKSearchResults` object.

Each `search.SearchResult` object may have iPython widgets attached to it, such
as label buttons. These are used for obtaining user-provided labels, etc.
