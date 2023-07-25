# coding=utf-8
# Copyright 2023 The Chirp Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementations of inference interfaces for applying trained models."""

import dataclasses
import tempfile
from typing import Any

from absl import logging
from chirp.inference import interface
from chirp.models import frontend
from chirp.models import handcrafted_features
from chirp.taxonomy import namespace
from chirp.taxonomy import namespace_db
from etils import epath
from ml_collections import config_dict
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_hub as hub


def model_class_map() -> dict[str, Any]:
  """Get the mapping of model keys to classes."""
  return {
      'taxonomy_model_tf': TaxonomyModelTF,
      'separator_model_tf': SeparatorModelTF,
      'birb_separator_model_tf1': BirbSepModelTF1,
      'birdnet': BirdNet,
      'placeholder_model': PlaceholderModel,
      'separate_embed_model': SeparateEmbedModel,
      'tfhub_model': TFHubModel,
  }


@dataclasses.dataclass
class SeparateEmbedModel(interface.EmbeddingModel):
  """Wrapper for separate separation and embedding models.

  Note: Use the separation model's sample rate. The embedding model's sample
  rate is used to resample prior to computing the embedding.

  Attributes:
    taxonomy_model_tf_config: Configuration for a TaxonomyModelTF.
    separator_model_tf_config: Configuration for a SeparationModelTF.
    embed_raw: If True, the outputs will include embeddings of the original
      audio in addition to embeddings for the separated channels. The embeddings
      will have shape [T, C+1, D], with the raw audio embedding on channel 0.
    separation_model: SeparationModelTF, automatically populated during init.
    embedding_model: TaxonomyModelTF, automatically populated during init.
  """

  taxonomy_model_tf_config: config_dict.ConfigDict
  separator_model_tf_config: config_dict.ConfigDict
  embed_raw: bool = True

  # Populated during init.
  separation_model: Any = None
  embedding_model: Any = None

  def __post_init__(self):
    if self.separation_model is None:
      self.separation_model = SeparatorModelTF(**self.separator_model_tf_config)
      self.embedding_model = TaxonomyModelTF(**self.taxonomy_model_tf_config)
    if self.separation_model.sample_rate != self.embedding_model.sample_rate:
      raise ValueError(
          'Separation and embedding models must have matching rates.'
      )

  def embed(self, audio_array: np.ndarray) -> interface.InferenceOutputs:
    # Frame the audio according to the embedding model's config.
    # We then apply separation to each frame independently, and embed
    # the separated audio.
    framed_audio = self.frame_audio(
        audio_array,
        self.embedding_model.window_size_s,
        self.embedding_model.hop_size_s,
    )
    # framed_audio has shape [Frames, Time]
    separation_outputs = self.separation_model.batch_embed(framed_audio)
    # separated_audio has shape [F, C, T]
    separated_audio = separation_outputs.separated_audio

    if self.embed_raw:
      separated_audio = np.concatenate(
          [
              framed_audio[:, np.newaxis, : separated_audio.shape[-1]],
              separated_audio,
          ],
          axis=1,
      )
    num_frames = separated_audio.shape[0]
    num_channels = separated_audio.shape[1]
    num_samples = separated_audio.shape[2]
    separated_audio = np.reshape(separated_audio, [-1, num_samples])

    embedding_outputs = self.embedding_model.batch_embed(separated_audio)

    # Batch embeddings have shape [Batch, Time, Channels, Features]
    # Time is 1 because we have framed using the embedding model's window_size.
    # The batch size is num_frames * num_channels.
    embeddings = np.reshape(
        embedding_outputs.embeddings, [num_frames, num_channels, -1]
    )

    # Take the maximum logits over the channels dimension.
    if embedding_outputs.logits is not None:
      max_logits = {}
      for k, v in embedding_outputs.logits.items():
        v = v.reshape([num_frames, num_channels, -1])
        max_logits[k] = np.max(v, axis=1)
    else:
      max_logits = None

    return interface.InferenceOutputs(
        embeddings=embeddings,
        logits=max_logits,
        # Because the separated audio is framed, it does not match the
        # outputs interface, so we do not return it.
        separated_audio=None,
    )


@dataclasses.dataclass
class BirbSepModelTF1(interface.EmbeddingModel):
  """Separation model from the Bird MixIT paper."""

  model_path: str
  window_size_s: float
  keep_raw_channel: bool

  # The following are populated at init time.
  session: Any | None = None
  input_placeholder_ns: Any | None = None
  output_tensor_ns: Any | None = None

  def __post_init__(self):
    """Load model files and create TF1 session graph."""
    metagraph_path_ns = epath.Path(self.model_path) / 'inference.meta'
    checkpoint_path = tf.train.latest_checkpoint(self.model_path)
    graph_ns = tf.Graph()
    sess_ns = tf1.Session(graph=graph_ns)
    with graph_ns.as_default():
      new_saver = tf1.train.import_meta_graph(metagraph_path_ns)
      new_saver.restore(sess_ns, checkpoint_path)
      self.input_placeholder_ns = graph_ns.get_tensor_by_name(
          'input_audio/receiver_audio:0'
      )
      self.output_tensor_ns = graph_ns.get_tensor_by_name(
          'denoised_waveforms:0'
      )
    self.session = sess_ns

  def embed(self, audio_array: Any) -> interface.InferenceOutputs:
    start_sample = 0
    window_size = int(self.window_size_s * self.sample_rate)
    sep_chunks = []
    raw_chunks = []
    while start_sample <= audio_array.shape[0]:
      audio_chunk = audio_array[start_sample : start_sample + window_size]
      raw_chunks.append(audio_chunk)
      separated_audio = self.session.run(
          self.output_tensor_ns,
          feed_dict={
              self.input_placeholder_ns: audio_chunk[np.newaxis, np.newaxis, :]
          },
      )
      # Drop the extraneous batch dimension.
      separated_audio = np.squeeze(separated_audio, axis=0)
      sep_chunks.append(separated_audio)
      start_sample += window_size

    raw_chunks = np.concatenate(raw_chunks, axis=0)
    sep_chunks = np.concatenate(sep_chunks, axis=-1)
    if self.keep_raw_channel:
      sep_chunks = np.concatenate(
          [sep_chunks, raw_chunks[np.newaxis, :]], axis=0
      )
    return interface.InferenceOutputs(separated_audio=sep_chunks)

  def batch_embed(self, audio_batch: np.ndarray) -> interface.InferenceOutputs:
    return interface.batch_embed_from_embed_fn(self.embed, audio_batch)


@dataclasses.dataclass
class TaxonomyModelTF(interface.EmbeddingModel):
  """Taxonomy SavedModel.

  Attributes:
    model_path: Path to model files.
    window_size_s: Window size for framing audio in seconds. TODO(tomdenton):
      Ideally this should come from a model metadata file.
    hop_size_s: Hop size for inference.
    target_class_list: If provided, restricts logits to this ClassList.
    model: Loaded TF SavedModel.
    class_list: Loaded class_list for the model's output logits.
    batchable: Whether the model supports batched input.
  """

  model_path: str
  window_size_s: float
  hop_size_s: float
  target_class_list: namespace.ClassList | None = None
  target_peak: float | None = 0.25

  # The following are populated during init.
  model: Any | None = None  # TF SavedModel
  class_list: namespace.ClassList | None = None
  batchable: bool = False

  def __post_init__(self):
    logging.info('Loading taxonomy model...')

    base_path = epath.Path(self.model_path)
    if (base_path / 'saved_model.pb').exists() and (
        base_path / 'assets'
    ).exists():
      # This looks like a TFHub downloaded model.
      model_path = base_path
      label_csv_path = epath.Path(self.model_path) / 'assets' / 'label.csv'
    else:
      # Probably a savedmodel distributed directly.
      model_path = base_path / 'savedmodel'
      label_csv_path = base_path / 'label.csv'

    self.model = tf.saved_model.load(model_path)
    with label_csv_path.open('r') as f:
      self.class_list = namespace.ClassList.from_csv(f)

    # Check whether the model support polymorphic batch shape.
    sig = self.model.signatures['serving_default']
    self.batchable = sig.inputs[0].shape[0] is None

  def embed(self, audio_array: np.ndarray) -> interface.InferenceOutputs:
    if self.batchable:
      return interface.embed_from_batch_embed_fn(self.batch_embed, audio_array)

    # Process one example at a time.
    # This should be fine on CPU, but may be somewhat inefficient for large
    # arrays on GPU or TPU.
    framed_audio = self.frame_audio(
        audio_array, self.window_size_s, self.hop_size_s
    )
    if self.target_peak is not None:
      framed_audio = self.normalize_audio(framed_audio, self.target_peak)
    all_logits, all_embeddings = self.model.infer_tf(framed_audio[:1])
    for window in framed_audio[1:]:
      logits, embeddings = self.model.infer_tf(window[np.newaxis, :])
      all_logits = np.concatenate([all_logits, logits], axis=0)
      all_embeddings = np.concatenate([all_embeddings, embeddings], axis=0)
    all_embeddings = all_embeddings[:, np.newaxis, :]

    all_logits = self.convert_logits(
        all_logits, self.class_list, self.target_class_list
    )

    return interface.InferenceOutputs(
        all_embeddings, {'label': all_logits}, None
    )

  def batch_embed(
      self, audio_batch: np.ndarray[Any, Any]
  ) -> interface.InferenceOutputs:
    if not self.batchable:
      return interface.batch_embed_from_embed_fn(self.embed, audio_batch)

    framed_audio = self.frame_audio(
        audio_batch, self.window_size_s, self.hop_size_s
    )
    if self.target_peak is not None:
      framed_audio = self.normalize_audio(framed_audio, self.target_peak)

    rebatched_audio = framed_audio.reshape([-1, framed_audio.shape[-1]])
    logits, embeddings = self.model.infer_tf(rebatched_audio)
    logits = self.convert_logits(
        logits, self.class_list, self.target_class_list
    )
    logits = np.reshape(logits, framed_audio.shape[:2] + (logits.shape[-1],))
    embeddings = np.reshape(
        embeddings, framed_audio.shape[:2] + (embeddings.shape[-1],)
    )

    return interface.InferenceOutputs(embeddings, {'label': logits}, None)


@dataclasses.dataclass
class SeparatorModelTF(interface.EmbeddingModel):
  """Separator SavedModel.

  Attributes:
    model_path: Path to model files.
    frame_size: Audio frame size for separation model.
    windows_size_s: Window size for framing audio in samples. The audio will be
      chunked into frames of size window_size_s, which may help avoid memory
      blowouts.
    target_class_list: If provided, restricts logits to this ClassList.
    model: Loaded TF SavedModel.
    class_list: Loaded class_list for the model's output logits.
  """

  model_path: str
  frame_size: int
  window_size_s: float | None = None
  target_class_list: namespace.ClassList | None = None

  # The following are populated during init.
  model: Any | None = None  # TF SavedModel
  class_list: namespace.ClassList | None = None

  def __post_init__(self):
    logging.info('Loading taxonomy model...')
    self.model = tf.saved_model.load(epath.Path(self.model_path) / 'savedmodel')
    label_csv_path = epath.Path(self.model_path) / 'label.csv'
    with label_csv_path.open('r') as f:
      self.class_list = namespace.ClassList.from_csv(f)

  def embed(self, audio_array: np.ndarray) -> interface.InferenceOutputs:
    # Drop samples to allow reshaping to frame_size
    excess_samples = audio_array.shape[0] % self.frame_size
    if excess_samples > 0:
      audio_array = audio_array[:-excess_samples]
    framed_audio = self.frame_audio(
        audio_array, self.window_size_s, self.window_size_s
    )
    framed_audio = np.reshape(
        framed_audio,
        [
            framed_audio.shape[0],
            framed_audio.shape[1] // self.frame_size,
            self.frame_size,
        ],
    )

    sep_audio, all_logits, all_embeddings = self.model.infer_tf(
        framed_audio[:1]
    )
    for window in framed_audio[1:]:
      separated, logits, embeddings = self.model.infer_tf(window[np.newaxis, :])
      sep_audio = np.concatenate([sep_audio, separated], axis=0)
      all_logits = np.concatenate([all_logits, logits], axis=0)
      all_embeddings = np.concatenate([all_embeddings, embeddings], axis=0)
    all_embeddings = all_embeddings[:, np.newaxis, :]

    # Recombine batch and time dimensions.
    sep_audio = np.reshape(sep_audio, [-1, sep_audio.shape[-1]])
    all_logits = np.reshape(all_logits, [-1, all_logits.shape[-1]])
    all_logits = self.convert_logits(
        all_logits, self.class_list, self.target_class_list
    )
    all_embeddings = np.reshape(all_embeddings, [-1, all_embeddings.shape[-1]])
    return interface.InferenceOutputs(
        all_embeddings, {'label': all_logits}, sep_audio
    )

  def batch_embed(self, audio_batch: np.ndarray) -> interface.InferenceOutputs:
    return interface.batch_embed_from_embed_fn(self.embed, audio_batch)


@dataclasses.dataclass
class BirdNet(interface.EmbeddingModel):
  """Wrapper for BirdNet models.

  Attributes:
    model_path: Path to the saved model checkpoint or TFLite file.
    class_list_name: Name of the BirdNet class list.
    window_size_s: Window size for framing audio in samples.
    hop_size_s: Hop size for inference.
    num_tflite_threads: Number of threads to use with TFLite model.
    target_class_list: If provided, restricts logits to this ClassList.
    model: The TF SavedModel or TFLite interpreter.
    tflite: Whether the model is a TFLite model.
    class_list: The loaded class list.
  """

  model_path: str
  class_list_name: str = 'birdnet_v2_1'
  window_size_s: float = 3.0
  hop_size_s: float = 3.0
  num_tflite_threads: int = 16
  target_class_list: namespace.ClassList | None = None
  # The following are populated during init.
  model: Any | None = None
  tflite: bool = False
  class_list: namespace.ClassList | None = None

  def __post_init__(self):
    logging.info('Loading BirdNet model...')
    if self.model_path.endswith('.tflite'):
      self.tflite = True
      with tempfile.NamedTemporaryFile() as tmpf:
        model_file = epath.Path(self.model_path)
        model_file.copy(tmpf.name, overwrite=True)
        self.model = tf.lite.Interpreter(
            tmpf.name, num_threads=self.num_tflite_threads
        )
      self.model.allocate_tensors()
    else:
      self.tflite = False
      self.model = tf.saved_model.load(self.model_path)

  def embed_saved_model(
      self, audio_array: np.ndarray
  ) -> interface.InferenceOutputs:
    """Get logits using the BirdNet SavedModel."""
    # Note that there is no easy way to get the embedding from the SavedModel.
    all_logits = self.model(audio_array[:1])
    for window in audio_array[1:]:
      logits = self.model(window[np.newaxis, :])
      all_logits = np.concatenate([all_logits, logits], axis=0)
    all_logits = self.convert_logits(
        all_logits, self.class_list, self.target_class_list
    )
    return interface.InferenceOutputs(
        None, {self.class_list_name: all_logits}, None
    )

  def embed_tflite(self, audio_array: np.ndarray) -> interface.InferenceOutputs:
    """Create an embedding and logits using the BirdNet TFLite model."""
    input_details = self.model.get_input_details()[0]
    output_details = self.model.get_output_details()[0]
    embedding_idx = output_details['index'] - 1
    embeddings = []
    logits = []
    for audio in audio_array:
      self.model.set_tensor(
          input_details['index'], np.float32(audio)[np.newaxis, :]
      )
      self.model.invoke()
      logits.append(self.model.get_tensor(output_details['index']))
      embeddings.append(self.model.get_tensor(embedding_idx))
    # Create [Batch, 1, Features]
    embeddings = np.array(embeddings)
    logits = np.array(logits)
    logits = self.convert_logits(
        logits, self.class_list, self.target_class_list
    )
    return interface.InferenceOutputs(
        embeddings, {self.class_list_name: logits}, None
    )

  def embed(self, audio_array: np.ndarray) -> interface.InferenceOutputs:
    framed_audio = self.frame_audio(
        audio_array, self.window_size_s, self.hop_size_s
    )
    if self.tflite:
      return self.embed_tflite(framed_audio)
    else:
      return self.embed_saved_model(framed_audio)

  def batch_embed(self, audio_batch: np.ndarray) -> interface.InferenceOutputs:
    return interface.batch_embed_from_embed_fn(self.embed, audio_batch)


@dataclasses.dataclass
class HandcraftedFeaturesModel(interface.EmbeddingModel):
  """Wrapper for simple feature extraction."""

  window_size_s: float
  hop_size_s: float
  melspec_config: config_dict.ConfigDict
  features_config: config_dict.ConfigDict

  @classmethod
  def beans_baseline(cls, sample_rate=32000, frame_rate=100):
    stride = sample_rate // frame_rate
    mel_config = config_dict.ConfigDict({
        'sample_rate': sample_rate,
        'features': 160,
        'stride': stride,
        'kernel_size': 2 * stride,
        'freq_range': (60.0, sample_rate / 2.0),
        'scaling_config': frontend.LogScalingConfig(),
    })
    features_config = config_dict.ConfigDict({
        'compute_mfccs': True,
        'aggregation': 'beans',
    })
    # pylint: disable=unexpected-keyword-arg
    return HandcraftedFeaturesModel(
        sample_rate=sample_rate,
        window_size_s=1.0,
        hop_size_s=1.0,
        melspec_config=mel_config,
        features_config=features_config,
    )

  def __post_init__(self):
    self.melspec_layer = frontend.MelSpectrogram(**self.melspec_config)
    self.features_layer = handcrafted_features.HandcraftedFeatures(
        **self.features_config
    )

  def embed(self, audio_array: np.ndarray) -> interface.InferenceOutputs:
    framed_audio = self.frame_audio(
        audio_array, self.window_size_s, self.hop_size_s
    )
    melspec = self.melspec_layer.apply({}, framed_audio)
    features = self.features_layer.apply(
        {}, melspec[:, :, :, np.newaxis], train=False
    )
    # Add a trivial channels dimension.
    features = features[:, np.newaxis, :]
    return interface.InferenceOutputs(features, None, None)

  def batch_embed(self, audio_batch: np.ndarray) -> interface.InferenceOutputs:
    return interface.batch_embed_from_embed_fn(self.embed, audio_batch)


@dataclasses.dataclass
class TFHubModel(interface.EmbeddingModel):
  """Generic wrapper for TFHub models which produce embeddings."""

  model_url: str
  embedding_index: int
  logits_index: int = -1

  @classmethod
  def yamnet(cls):
    # Parent class takes a sample_rate arg which pylint doesn't find.
    # pylint: disable=too-many-function-args
    return TFHubModel(16000, 'https://tfhub.dev/google/yamnet/1', 1, 0)

  @classmethod
  def vggish(cls):
    # pylint: disable=too-many-function-args
    return TFHubModel(16000, 'https://tfhub.dev/google/vggish/1', -1)

  def __post_init__(self):
    self.model = hub.load(self.model_url)

  def embed(
      self, audio_array: np.ndarray[Any, np.dtype[Any]]
  ) -> interface.InferenceOutputs:
    outputs = self.model(audio_array)
    if self.embedding_index < 0:
      embeddings = outputs
    else:
      embeddings = outputs[self.embedding_index]
    if len(embeddings.shape) == 1:
      embeddings = embeddings[np.newaxis, :]
    elif len(embeddings.shape) != 2:
      raise ValueError('Embeddings should have shape [Depth] or [Time, Depth].')

    if self.logits_index >= 0:
      logits = {'label': outputs[self.logits_index]}
    else:
      logits = None
    embeddings = embeddings[:, np.newaxis, :]
    return interface.InferenceOutputs(embeddings, logits, None, False)

  def batch_embed(self, audio_batch: np.ndarray) -> interface.InferenceOutputs:
    return interface.batch_embed_from_embed_fn(self.embed, audio_batch)


@dataclasses.dataclass
class PlaceholderModel(interface.EmbeddingModel):
  """Test implementation of the EmbeddingModel interface."""

  embedding_size: int = 128
  make_embeddings: bool = True
  make_logits: bool = True
  make_separated_audio: bool = True
  target_class_list: namespace.ClassList | None = None
  window_size_s: float = 1.0
  hop_size_s: float = 1.0

  def __post_init__(self):
    db = namespace_db.load_db()
    self.class_list = db.class_lists['caples']

  def embed(self, audio_array: np.ndarray) -> interface.InferenceOutputs:
    outputs = {}
    time_size = audio_array.shape[0] // int(
        self.window_size_s * self.sample_rate
    )
    if self.make_embeddings:
      outputs['embeddings'] = np.zeros(
          [time_size, 1, self.embedding_size], np.float32
      )
    if self.make_logits:
      outputs['logits'] = {
          'label': np.zeros(
              [time_size, len(self.class_list.classes)], np.float32
          ),
      }
      outputs['logits']['label'] = self.convert_logits(
          outputs['logits']['label'], self.class_list, self.target_class_list
      )
    if self.make_separated_audio:
      outputs['separated_audio'] = np.zeros(
          [2, audio_array.shape[-1]], np.float32
      )
    return interface.InferenceOutputs(**outputs)

  def batch_embed(self, audio_batch: np.ndarray) -> interface.InferenceOutputs:
    return interface.batch_embed_from_embed_fn(self.embed, audio_batch)
