# Model Zoo for Bioacoustics

This package handles audio embedding models. We provide a simple interface
(`zoo_interface.EmbeddingModel`) for wrapping models which transform audio clips
(of any length) into embeddings. The common interface allows clean comparison
of models for evaluation purposes, and also allows users to freely choose the
most appropriate model for their work.

The most convenient way to load a predefined model is like so:
```m = model_configs.load_model_by_name('perch_8')```
which loads the Perch v8 model automatically from Kaggle Models. The set of
currently implemented models can be inspected in
`model_configs.ModelConfigName`.

## The Embedding Model Interface

We provide a model wrapping interface `zoo_interface.EmbeddingModel` which can
be implemented by a wide range of models providing some combination of
classification logits, embeddings, and separated audio. Implementations are
mostly provided in `models.py`, including:

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
audio array. The outputs are an `zoo_interface.InferenceOutputs` instance, which
contains optional embeddings, logits, and separated audio.
