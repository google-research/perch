# coding=utf-8
# Copyright 2022 The Chirp Authors.
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
from chirp.taxonomy import namespace
from chirp.taxonomy import namespace_db
from etils import epath
import numpy as np
import scipy
import tensorflow as tf
import tensorflow.compat.v1 as tf1


@dataclasses.dataclass
class SeparateEmbedModel(interface.EmbeddingModel):
  """Wrapper for separate separation and embedding models.

  Note: Use the separation model's sample rate. The embedding model's sample
  rate is used to resample prior to computing the embedding.
  """

  separation_model: interface.EmbeddingModel
  embedding_model: interface.EmbeddingModel

  def embed(self, audio_array: np.ndarray) -> interface.InferenceOutputs:
    # Apply the separation model.
    separation_outputs = self.separation_model.embed(audio_array)
    if self.separation_model.sample_rate != self.embedding_model.sample_rate:
      new_length = int(
          self.embedding_model.sample_rate
          / self.separation_model.sample_rate
          * separation_outputs.separated_audio.shape[-1]
      )
      separated_audio = scipy.signal.resample(
          separation_outputs.separated_audio, new_length, axis=-1
      )
    else:
      separated_audio = separation_outputs.separated_audio

    embedding_outputs = self.embedding_model.batch_embed(separated_audio)
    # embedding output has shape [C, T, 1, D]; rearrange to [T, C, D].
    embeddings = embedding_outputs.embeddings.swapaxes(0, 2).squeeze(0)

    # Take the maximum logits over the channels dimension.
    if embedding_outputs.logits is not None:
      max_logits = {}
      for k, v in embedding_outputs.logits.items():
        max_logits[k] = np.max(v, axis=0)
    else:
      max_logits = None

    return interface.InferenceOutputs(
        embeddings=embeddings,
        logits=max_logits,
        separated_audio=separation_outputs.separated_audio,
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
  """

  model_path: str
  window_size_s: float
  hop_size_s: float
  target_class_list: namespace.ClassList | None = None

  # The following are populated during init.
  model: Any | None = None  # TF SavedModel
  class_list: namespace.ClassList | None = None

  def __post_init__(self):
    logging.info('Loading taxonomy model...')
    self.model = tf.saved_model.load(epath.Path(self.model_path) / 'savedmodel')
    label_csv_path = epath.Path(self.model_path) / 'label.csv'
    with label_csv_path.open('r') as f:
      self.class_list = namespace.ClassList.from_csv('label', f)

  def embed(self, audio_array: np.ndarray) -> interface.InferenceOutputs:
    # Process one example at a time.
    # This should be fine on CPU, but may be somewhat inefficient for large
    # arrays on GPU or TPU.
    framed_audio = self.frame_audio(
        audio_array, self.window_size_s, self.hop_size_s
    )
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
        all_embeddings, {self.class_list.name: all_logits}, None
    )


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
      self.class_list = namespace.ClassList.from_csv('label', f)

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
        all_embeddings, {self.class_list.name: all_logits}, sep_audio
    )


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
    embeddings = np.array(embeddings)[:, np.newaxis, :]
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


@dataclasses.dataclass
class PlaceholderModel(interface.EmbeddingModel):
  """Test implementation of the EmbeddingModel interface."""

  embedding_size: int = 128
  make_embeddings: bool = True
  make_logits: bool = True
  make_separated_audio: bool = True
  target_class_list: namespace.ClassList | None = None

  def __post_init__(self):
    db = namespace_db.load_db()
    self.class_list = db.class_lists['caples']

  def embed(self, audio_array: np.ndarray) -> interface.InferenceOutputs:
    outputs = {}
    time_size = audio_array.shape[0] // self.sample_rate
    if self.make_embeddings:
      outputs['embeddings'] = np.zeros(
          [time_size, 1, self.embedding_size], np.float32
      )
    if self.make_logits:
      outputs['logits'] = {
          'label': np.zeros([time_size, self.class_list.size], np.float32),
      }
      outputs['logits']['label'] = self.convert_logits(
          outputs['logits']['label'], self.class_list, self.target_class_list
      )
    if self.make_separated_audio:
      outputs['separated_audio'] = np.zeros(
          [2, audio_array.shape[-1]], np.float32
      )
    return interface.InferenceOutputs(**outputs)
