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

Step-by-step:

* Run the main repository's installation instructions.

* Download and extract the Perch model from TFHub:
  https://tfhub.dev/google/bird-vocalization-classifier/2

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
