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
    window_size: Window size for framing audio in samples. TODO(tomdenton):
      Ideally this should come from a model metadata file.
    model: Loaded TF SavedModel.
    class_list: Loaded class_list for the model's output logits.
  """
  model_path: str
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
    all_logits, all_embeddings = self.model.infer_tf(audio_array[:1])
    for window in audio_array[1:]:
      logits, embeddings = self.model.infer_tf(window[np.newaxis, :])
      all_logits = np.concatenate([all_logits, logits], axis=0)
      all_embeddings = np.concatenate([all_embeddings, embeddings], axis=0)
    return interface.InferenceOutputs(all_embeddings,
                                      {self.class_list.name: all_logits}, None)


@dataclasses.dataclass
class BirdNet(interface.EmbeddingModel):
  """Wrapper for BirdNet models.

  Attributes:
    model_path: Path to the saved model checkpoint or TFLite file.
    class_list_name: Name of the BirdNet class list.
    num_tflite_threads: Number of threads to use with TFLite model.
    model: The TF SavedModel or TFLite interpreter.
    tflite: Whether the model is a TFLite model.
    class_list: The loaded class list.
  """
  model_path: str
  class_list_name: str = 'birdnet_v2_1'
  num_tflite_threads: int = 16
  # The following are populated during init.
  model: Optional[Any] = None  #
  tflite: bool = False
  class_list: Optional[namespace.ClassList] = None

  def __post_init__(self):
    logging.info('Loading BirdNet model...')
    if self.model_path.endswith('.tflite'):
      self.tflite = True
      self.model = tf.lite.Interpreter(
          self.model_path, num_threads=self.num_tflite_threads)
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
      self.model.set_tensor(input_details['index'], np.float32(audio))
      self.model.invoke()
      logits.append(self.model.get_tensor(output_details['index']))
      embeddings.append(self.model.get_tensor(embedding_idx))
    embeddings = np.array(embeddings)
    logits = np.array(logits)
    return interface.InferenceOutputs(embeddings,
                                      {self.class_list_name: logits}, None)

  def embed(self, audio_array: np.ndarray) -> interface.InferenceOutputs:
    if self.tflite:
      return self.embed_tflite(audio_array)
    else:
      return self.embed_saved_model(audio_array)


@dataclasses.dataclass
class DummyModel(interface.EmbeddingModel):
  # For test runs only.
  embedding_size: int = 128
  window_size_s = -1

  def embed(self, audio_array: np.ndarray) -> interface.InferenceOutputs:
    return interface.InferenceOutputs(
        embeddings=np.zeros(
            [audio_array.shape[1] //
             self.sample_rate, self.embedding_size], np.float32))
