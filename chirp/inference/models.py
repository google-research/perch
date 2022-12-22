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
from typing import Any, Optional

from absl import logging
from chirp.inference import interface
from chirp.taxonomy import namespace
from etils import epath
import numpy as np
import tensorflow as tf


@dataclasses.dataclass
class TaxonomyModelTF(interface.EmbeddingModel):
  """Taxonomy SavedModel.

  Attributes:
    model_path: Path to model files.
    window_size_s: Window size for framing audio in seconds. TODO(tomdenton):
      Ideally this should come from a model metadata file.
    hop_size_s: Hop size for inference.
    model: Loaded TF SavedModel.
    class_list: Loaded class_list for the model's output logits.
  """
  model_path: str
  window_size_s: float
  hop_size_s: float

  # The following are populated during init.
  model: Optional[Any] = None  # TF SavedModel
  class_list: Optional[namespace.ClassList] = None

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
    framed_audio = self.frame_audio(audio_array, self.window_size_s,
                                    self.hop_size_s)
    all_logits, all_embeddings = self.model.infer_tf(framed_audio[:1])
    for window in framed_audio[1:]:
      logits, embeddings = self.model.infer_tf(window[np.newaxis, :])
      all_logits = np.concatenate([all_logits, logits], axis=0)
      all_embeddings = np.concatenate([all_embeddings, embeddings], axis=0)
    all_embeddings = all_embeddings[:, np.newaxis, :]
    return interface.InferenceOutputs(all_embeddings,
                                      {self.class_list.name: all_logits}, None)


@dataclasses.dataclass
class SeparatorModelTF(interface.EmbeddingModel):
  """Separator SavedModel.

  Attributes:
    model_path: Path to model files.
    frame_size: Audio frame size for separation model.
    windows_size_s: Window size for framing audio in samples. The audio will be
      chunked into frames of size window_size_s, which may help avoid memory
      blowouts.
    model: Loaded TF SavedModel.
    class_list: Loaded class_list for the model's output logits.
  """
  model_path: str
  frame_size: int
  window_size_s: Optional[float] = None
  # The following are populated during init.
  model: Optional[Any] = None  # TF SavedModel
  class_list: Optional[namespace.ClassList] = None

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
    framed_audio = self.frame_audio(audio_array, self.window_size_s,
                                    self.window_size_s)
    framed_audio = np.reshape(framed_audio, [
        framed_audio.shape[0], framed_audio.shape[1] // self.frame_size,
        self.frame_size
    ])

    sep_audio, all_logits, all_embeddings = self.model.infer_tf(
        framed_audio[:1])
    for window in framed_audio[1:]:
      separated, logits, embeddings = self.model.infer_tf(window[np.newaxis, :])
      sep_audio = np.concatenate([sep_audio, separated], axis=0)
      all_logits = np.concatenate([all_logits, logits], axis=0)
      all_embeddings = np.concatenate([all_embeddings, embeddings], axis=0)
    all_embeddings = all_embeddings[:, np.newaxis, :]

    # Recombine batch and time dimensions.
    sep_audio = np.reshape(sep_audio, [-1, sep_audio.shape[-1]])
    all_logits = np.reshape(all_logits, [-1, all_logits.shape[-1]])
    all_embeddings = np.reshape(all_embeddings, [-1, all_embeddings.shape[-1]])
    return interface.InferenceOutputs(all_embeddings,
                                      {self.class_list.name: all_logits},
                                      sep_audio)


@dataclasses.dataclass
class BirdNet(interface.EmbeddingModel):
  """Wrapper for BirdNet models.

  Attributes:
    model_path: Path to the saved model checkpoint or TFLite file.
    class_list_name: Name of the BirdNet class list.
    window_size_s: Window size for framing audio in samples.
    hop_size_s: Hop size for inference.
    num_tflite_threads: Number of threads to use with TFLite model.
    model: The TF SavedModel or TFLite interpreter.
    tflite: Whether the model is a TFLite model.
    class_list: The loaded class list.
  """
  model_path: str
  class_list_name: str = 'birdnet_v2_1'
  window_size_s: float = 3.0
  hop_size_s: float = 3.0
  num_tflite_threads: int = 16
  # The following are populated during init.
  model: Optional[Any] = None
  tflite: bool = False
  class_list: Optional[namespace.ClassList] = None

  def __post_init__(self):
    logging.info('Loading BirdNet model...')
    if self.model_path.endswith('.tflite'):
      self.tflite = True
      with tempfile.NamedTemporaryFile() as tmpf:
        model_file = epath.Path(self.model_path)
        model_file.copy(tmpf.name, overwrite=True)
        self.model = tf.lite.Interpreter(
            tmpf.name, num_threads=self.num_tflite_threads)
      self.model.allocate_tensors()
    else:
      self.tflite = False
      self.model = tf.saved_model.load(self.model_path)

  def embed_saved_model(self,
                        audio_array: np.ndarray) -> interface.InferenceOutputs:
    """Get logits using the BirdNet SavedModel."""
    # Note that there is no easy way to get the embedding from the SavedModel.
    all_logits = self.model(audio_array[:1])
    for window in audio_array[1:]:
      logits = self.model(window[np.newaxis, :])
      all_logits = np.concatenate([all_logits, logits], axis=0)
    return interface.InferenceOutputs(None, {self.class_list_name: all_logits},
                                      None)

  def embed_tflite(self, audio_array: np.ndarray) -> interface.InferenceOutputs:
    """Create an embedding and logits using the BirdNet TFLite model."""
    input_details = self.model.get_input_details()[0]
    output_details = self.model.get_output_details()[0]
    embedding_idx = output_details['index'] - 1
    embeddings = []
    logits = []
    for audio in audio_array:
      self.model.set_tensor(input_details['index'],
                            np.float32(audio)[np.newaxis, :])
      self.model.invoke()
      logits.append(self.model.get_tensor(output_details['index']))
      embeddings.append(self.model.get_tensor(embedding_idx))
    # Create [Batch, 1, Features]
    embeddings = np.array(embeddings)[:, np.newaxis, :]
    logits = np.array(logits)
    return interface.InferenceOutputs(embeddings,
                                      {self.class_list_name: logits}, None)

  def embed(self, audio_array: np.ndarray) -> interface.InferenceOutputs:
    framed_audio = self.frame_audio(audio_array, self.window_size_s,
                                    self.hop_size_s)
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

  def embed(self, audio_array: np.ndarray) -> interface.InferenceOutputs:
    outputs = {}
    if self.make_embeddings:
      outputs['embeddings'] = np.zeros(
          [audio_array.shape[0] // self.sample_rate, self.embedding_size],
          np.float32)
    if self.make_logits:
      outputs['logits'] = {
          'label': np.zeros([10], np.float32),
      }
    if self.make_separated_audio:
      outputs['separated_audio'] = np.zeros([2, audio_array.shape[-1]],
                                            np.float32)
    return interface.InferenceOutputs(**outputs)
