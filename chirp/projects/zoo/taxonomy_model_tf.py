# coding=utf-8
# Copyright 2024 The Perch Authors.
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

"""Perch Taxonomy Model class."""

import dataclasses
from typing import Any

from absl import logging
from chirp.projects.zoo import zoo_interface
from chirp.taxonomy import namespace
from etils import epath
from ml_collections import config_dict
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


PERCH_TF_HUB_URL = (
    'https://www.kaggle.com/models/google/'
    'bird-vocalization-classifier/frameworks/TensorFlow2/'
    'variations/bird-vocalization-classifier/versions'
)

SURFPERCH_TF_HUB_URL = (
    'https://www.kaggle.com/models/google/surfperch/TensorFlow2/1'
)


@dataclasses.dataclass
class TaxonomyModelTF(zoo_interface.EmbeddingModel):
  """Taxonomy SavedModel.

  Attributes:
    model_path: Path to model files.
    window_size_s: Window size for framing audio in seconds. TODO(tomdenton):
      Ideally this should come from a model metadata file.
    hop_size_s: Hop size for inference.
    model: Loaded TF SavedModel.
    class_list: Loaded class_list for the model's output logits.
    batchable: Whether the model supports batched input.
    target_peak: Peak normalization value.
  """

  model_path: str
  window_size_s: float
  hop_size_s: float
  model: Any  # TF SavedModel
  class_list: dict[str, namespace.ClassList]
  batchable: bool
  target_peak: float | None = 0.25
  tfhub_version: int | None = None

  @classmethod
  def is_batchable(cls, model: Any) -> bool:
    sig = model.signatures['serving_default']
    return sig.inputs[0].shape[0] is None

  @classmethod
  def load_class_lists(cls, csv_glob):
    class_lists = {}
    for csv_path in csv_glob:
      with csv_path.open('r') as f:
        key = csv_path.name.replace('.csv', '')
        class_lists[key] = namespace.ClassList.from_csv(f)
    return class_lists

  @classmethod
  def from_tfhub(cls, config: config_dict.ConfigDict) -> 'TaxonomyModelTF':
    if not hasattr(config, 'tfhub_version') or config.tfhub_version is None:
      raise ValueError('tfhub_version is required to load from TFHub.')
    if config.model_path:
      raise ValueError(
          'Exactly one of tfhub_version and model_path should be set.'
      )
    if hasattr(config, 'tfhub_path'):
      tfhub_path = config.tfhub_path
      del config.tfhub_path
    else:
      tfhub_path = PERCH_TF_HUB_URL

    if tfhub_path == PERCH_TF_HUB_URL and config.tfhub_version in (5, 6, 7):
      # Due to SNAFUs uploading the new model version to KaggleModels,
      # some version numbers were skipped.
      raise ValueError('TFHub version 5, 6, and 7 do not exist.')

    model_url = f'{tfhub_path}/{config.tfhub_version}'
    # This model behaves exactly like the usual saved_model.
    model = hub.load(model_url)

    # Check whether the model support polymorphic batch shape.
    batchable = cls.is_batchable(model)

    # Get the labels CSV from TFHub.
    model_path = hub.resolve(model_url)
    config.model_path = model_path
    class_lists_glob = (epath.Path(model_path) / 'assets').glob('*.csv')
    class_lists = cls.load_class_lists(class_lists_glob)
    return cls(
        model=model,
        class_list=class_lists,
        batchable=batchable,
        **config,
    )

  @classmethod
  def load_version(
      cls, tfhub_version: int, hop_size_s: float = 5.0
  ) -> 'TaxonomyModelTF':
    cfg = config_dict.ConfigDict({
        'model_path': '',
        'sample_rate': 32000,
        'window_size_s': 5.0,
        'hop_size_s': hop_size_s,
        'target_peak': 0.25,
        'tfhub_version': tfhub_version,
    })
    return cls.from_tfhub(cfg)

  @classmethod
  def load_surfperch_version(
      cls, tfhub_version: int, hop_size_s: float = 5.0
  ) -> 'TaxonomyModelTF':
    """Load a model from TFHub."""
    cfg = config_dict.ConfigDict({
        'model_path': '',
        'sample_rate': 32000,
        'window_size_s': 5.0,
        'hop_size_s': hop_size_s,
        'target_peak': 0.25,
        'tfhub_version': tfhub_version,
        'tfhub_path': SURFPERCH_TF_HUB_URL,
    })
    return cls.from_tfhub(cfg)

  @classmethod
  def from_config(cls, config: config_dict.ConfigDict) -> 'TaxonomyModelTF':
    logging.info('Loading taxonomy model...')

    if hasattr(config, 'tfhub_version') and config.tfhub_version is not None:
      return cls.from_tfhub(config)

    base_path = epath.Path(config.model_path)
    if (base_path / 'saved_model.pb').exists() and (
        base_path / 'assets'
    ).exists():
      # This looks like a downloaded TFHub model.
      model_path = base_path
      class_lists_glob = (epath.Path(model_path) / 'assets').glob('*.csv')
    else:
      # Probably a savedmodel distributed directly.
      model_path = base_path / 'savedmodel'
      class_lists_glob = epath.Path(base_path).glob('*.csv')

    model = tf.saved_model.load(model_path)
    class_lists = cls.load_class_lists(class_lists_glob)

    # Check whether the model support polymorphic batch shape.
    batchable = cls.is_batchable(model)
    return cls(
        model=model, class_list=class_lists, batchable=batchable, **config
    )

  def get_classifier_head(self, classes: list[str]):
    """Extract a classifier head for the desired subset of classes."""
    if self.tfhub_version is not None:
      # This is a model loaded from TFHub.
      # We need to extract the weights and biases from the saved model.
      vars_filepath = f'{self.model_path}/variables/variables'
    else:
      vars_filepath = f'{self.model_path}/savedmodel/variables/variables'

    def _get_weights_and_bias(num_classes: int):
      weights = None
      bias = None
      for vname, vshape in tf.train.list_variables(vars_filepath):
        if len(vshape) == 1 and vshape[-1] == num_classes:
          if bias is None:
            bias = tf.train.load_variable(vars_filepath, vname)
          else:
            raise ValueError('Multiple possible biases for class list.')
        if len(vshape) == 2 and vshape[-1] == num_classes:
          if weights is None:
            weights = tf.train.load_variable(vars_filepath, vname)
          else:
            raise ValueError('Multiple possible weights for class list.')
      if hasattr(weights, 'numpy'):
        weights = weights.numpy()
      if hasattr(bias, 'numpy'):
        bias = bias.numpy()
      return weights, bias

    class_wts = {}
    for logit_key in self.class_list:
      num_classes = len(self.class_list[logit_key].classes)
      weights, bias = _get_weights_and_bias(num_classes)
      if weights is None or bias is None:
        raise ValueError(
            f'No weights or bias found for {logit_key} {num_classes}'
        )
      for i, k in enumerate(self.class_list[logit_key].classes):
        class_wts[k] = weights[:, i], bias[i]

    wts = []
    biases = []
    found_classes = []
    for target_class in classes:
      if target_class not in class_wts:
        continue
      wts.append(class_wts[target_class][0])
      biases.append(class_wts[target_class][1])
      found_classes.append(target_class)
    print(f'Found classes: {found_classes}')
    return found_classes, np.stack(wts, axis=-1), np.stack(biases, axis=-1)

  def embed(self, audio_array: np.ndarray) -> zoo_interface.InferenceOutputs:
    return zoo_interface.embed_from_batch_embed_fn(
        self.batch_embed, audio_array
    )

  def _nonbatchable_batch_embed(self, audio_batch: np.ndarray):
    """Embed a batch of audio with an old non-batchable model."""
    all_logits = []
    all_embeddings = []
    for audio in audio_batch:
      outputs = self.model.infer_tf(audio[np.newaxis, :])
      if hasattr(outputs, 'keys'):
        embedding = outputs.pop('embedding')
        logits = outputs.pop('label')
      else:
        # Assume the model output is always a (logits, embedding) twople.
        logits, embedding = outputs
      all_logits.append(logits)
      all_embeddings.append(embedding)
    all_logits = np.stack(all_logits, axis=0)
    all_embeddings = np.stack(all_embeddings, axis=0)
    return {
        'embedding': all_embeddings,
        'label': all_logits,
    }

  def batch_embed(
      self, audio_batch: np.ndarray[Any, Any]
  ) -> zoo_interface.InferenceOutputs:
    framed_audio = self.frame_audio(
        audio_batch, self.window_size_s, self.hop_size_s
    )
    framed_audio = self.normalize_audio(framed_audio, self.target_peak)
    rebatched_audio = framed_audio.reshape([-1, framed_audio.shape[-1]])

    if not self.batchable:
      outputs = self._nonbatchable_batch_embed(rebatched_audio)
    else:
      outputs = self.model.infer_tf(rebatched_audio)

    frontend_output = None
    if hasattr(outputs, 'keys'):
      # Dictionary-type outputs. Arrange appropriately.
      embeddings = outputs.pop('embedding')
      if 'frontend' in outputs:
        frontend_output = outputs.pop('frontend')
      # Assume remaining outputs are all logits.
      logits = outputs
    elif len(outputs) == 2:
      # Assume logits, embeddings outputs.
      label_logits, embeddings = outputs
      logits = {'label': label_logits}
    else:
      raise ValueError('Unexpected outputs type.')

    for k, v in logits.items():
      logits[k] = np.reshape(v, framed_audio.shape[:2] + (v.shape[-1],))
    # Unbatch and add channel dimension.
    embeddings = np.reshape(
        embeddings,
        framed_audio.shape[:2]
        + (
            1,
            embeddings.shape[-1],
        ),
    )
    return zoo_interface.InferenceOutputs(
        embeddings=embeddings,
        logits=logits,
        separated_audio=None,
        batched=True,
        frontend=frontend_output,
    )
