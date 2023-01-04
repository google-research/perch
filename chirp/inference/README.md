# Chirp Inference

This library is for applying trained models to data.

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
