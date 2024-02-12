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

"""Model callbacks library."""

import dataclasses
from typing import cast, Sequence

from absl import logging
from chirp.eval import eval_lib
from chirp.inference import interface
from chirp.inference import models as inference_models
from chirp.taxonomy import namespace
from chirp.taxonomy import namespace_db
from chirp.train import classifier
from chirp.train import hubert
from chirp.train import separator
from clu import checkpoint
from etils import epath
import jax
from jax import numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf

ConfigDict = ml_collections.ConfigDict


def pmap_with_remainder(
    model_callable: eval_lib.EvalModelCallable,
) -> eval_lib.EvalModelCallable:
  """Run a model callback in a multi-device setting.

  Since the model can be called with a variable batch size, this has to be
  handled in a multi-device environment. We do this by splitting the batch
  across the devices and hosts using `pmap`. If there is a remainder to the
  batch, then we process this in a separate call which is done on each host.

  Args:
    model_callable: A model callable (must be a JAX function).

  Returns:
    A model callable with the same signature but which uses data parallelism.
  """
  model_callable_pmap = jax.pmap(model_callable)
  model_callable_jit = jax.jit(model_callable)

  def parallel_model_callable(inputs: np.ndarray) -> np.ndarray:
    # Split the batch across devices
    n, m = jax.local_device_count(), inputs.shape[0]

    if m < n:
      return model_callable_jit(inputs)

    batch = jnp.reshape(inputs[: n * (m // n)], (n, m // n) + inputs.shape[1:])
    outputs = model_callable_pmap(batch)
    outputs = jnp.reshape(outputs, (n * (m // n),) + outputs.shape[2:])

    # Check if there is a remainder to the batch
    r = m - n * (m // n)
    if r == 0:
      return outputs  # pytype: disable=bad-return-type  # jnp-type
    else:
      # If not, run the remainder of the batch on each host
      batch = inputs[n * (m // n) :]
      remainder = model_callable_jit(batch)
      return jnp.concatenate([outputs, remainder])  # pytype: disable=bad-return-type  # jnp-type

  return parallel_model_callable


@dataclasses.dataclass
class TaxonomyModelCallback:
  """A model callback implementation for TaxonomyModel checkpoints.

  Attributes:
    init_config: TaxonomyModel configuration.
    workdir: path to the model checkpoint.
    use_learned_representations: If True, use the model's output weights as a
      learned representation for species seen during training. If False, reverts
      to the default behavior of using all embedded upstream recordings for
      artificially rare species to form search queries.
    learned_representation_blocklist: Species codes for learned representations
      which should *not* appear in the `learned_representations` mapping. This
      is analogous in result to having an allowlist for which species codes use
      the `learned_representations`. By default, this is set to False so that
      all eval sets use embedded class representatives with which to form
      species queries.
    model_callback: the fprop function used as part of the model callback,
      created automatically post-initialization.
    learned_representations: mapping from class name to its learned
      representation, created automatically post-initialization. If
      `use_learned_representations` is False, it is left empty, which results in
      the evaluation protocol relying instead on embedded upstream recordings to
      form search queries.
  """

  init_config: ConfigDict
  workdir: str
  use_learned_representations: bool = False
  learned_representation_blocklist: Sequence[str] = dataclasses.field(
      default_factory=list
  )
  # The following are populated during init.
  model_callback: eval_lib.EvalModelCallable = dataclasses.field(init=False)
  learned_representations: dict[str, np.ndarray] = dataclasses.field(
      init=False, default_factory=dict
  )

  def __post_init__(self):
    model_bundle, train_state = classifier.initialize_model(
        workdir=self.workdir, **self.init_config
    )
    # All hosts should load the same checkpoint
    multihost_ckpt = cast(checkpoint.MultihostCheckpoint, model_bundle.ckpt)
    ckpt = checkpoint.Checkpoint(multihost_ckpt.multihost_base_directory + '-0')
    train_state = ckpt.restore(train_state)
    variables = {'params': train_state.params, **train_state.model_state}

    def fprop(inputs):
      return model_bundle.model.apply(variables, inputs, train=False).embedding

    self.model_callback = pmap_with_remainder(fprop)

    if self.use_learned_representations:
      class_lists = {
          md.key: md.class_list for md in self.init_config.output_head_metadatas
      }
      class_list = class_lists['label']

      head_index = list(model_bundle.model.num_classes.keys()).index('label')
      output_weights = train_state.params[f'Dense_{head_index}']['kernel'].T
      self.learned_representations.update(
          {
              n: w
              for n, w in zip(class_list.classes, output_weights)
              if n not in self.learned_representation_blocklist
          }
      )

  def __call__(self, inputs: np.ndarray) -> np.ndarray:
    return np.asarray(self.model_callback(inputs))


@dataclasses.dataclass
class SeparatorTFCallback:
  """An eval model callback the embedding from an audio separator."""

  model_path: str
  use_learned_representations: bool = False
  learned_representation_blocklist: Sequence[str] = dataclasses.field(
      default_factory=list
  )
  frame_size: int = 32000
  # The following are populated during init.
  model_callback: eval_lib.EvalModelCallable = dataclasses.field(init=False)
  learned_representations: dict[str, np.ndarray] = dataclasses.field(
      init=False, default_factory=dict
  )

  def _load_learned_representations(self):
    """Loads classifier output weights from the separator."""
    label_csv_path = epath.Path(self.model_path) / 'label.csv'
    with label_csv_path.open('r') as f:
      class_list = namespace.ClassList.from_csv(f)
    # Load the output layer weights.
    variables_path = (
        epath.Path(self.model_path) / 'savedmodel/variables/variables'
    ).as_posix()
    variables = tf.train.list_variables(variables_path)
    candidates = []
    for v, v_shape in variables:
      # The classifier output layer is a 1D convolution with kernel size
      # (1, embedding_dim, num_classes).
      if (
          len(v_shape) == 3
          and v_shape[0] == 1
          and v_shape[-1] == len(class_list.classes)
      ):
        candidates.append(v)
    if not candidates:
      raise ValueError('Could not locate output weights layer.')
    elif len(candidates) > 1:
      raise ValueError(
          'Found multiple layers which could be the output weights layer (%s).'
          % candidates
      )
    else:
      output_weights = tf.train.load_variable(variables_path, candidates[0])
      output_weights = np.squeeze(output_weights)
    self.learned_representations.update(
        {
            n: w
            for n, w in zip(class_list.classes, output_weights)
            if n not in self.learned_representation_blocklist
        }
    )

  def __post_init__(self):
    logging.info('Loading separation model...')
    separation_model = tf.saved_model.load(
        epath.Path(self.model_path) / 'savedmodel'
    )

    def fprop(inputs):
      framed_inputs = np.reshape(
          inputs,
          [
              inputs.shape[0],
              inputs.shape[1] // self.frame_size,
              self.frame_size,
          ],
      )
      # Outputs are separated audio, logits, and embeddings.
      _, _, embeddings = separation_model.infer_tf(framed_inputs)
      # Embeddings have shape [B, T, D]; we need to aggregate over time.
      # For separation models, the mid-point embedding is usually best.
      midpt = embeddings.shape[1] // 2
      embeddings = embeddings[:, midpt, :]
      return embeddings

    self.model_callback = fprop
    if self.use_learned_representations:
      logging.info('Loading learned representations...')
      self._load_learned_representations()
    logging.info('Model loaded.')

  def __call__(self, inputs: np.ndarray) -> np.ndarray:
    return np.asarray(self.model_callback(inputs))


@dataclasses.dataclass
class EmbeddingModelCallback:
  """A general callback implementation for inference.EmbeddingModel wrappers.

  Attributes:
    model_key: Key for the model. See chirp.inference.models.model_class_map.
    model_config: Config dict for the target model.
    time_pooling: Named method for reducing embeddings over the time dimension.
      See chirp.inference.interface.InferenceOutputs.pooled_embeddings.
    channel_pooling: Named method for reducing embeddings channel dimension. See
      chirp.inference.interface.InferenceOutputs.pooled_embeddings.
    loaded_model: The instantiated interface.EmbeddingModel.
    model_callback: Eval callback.
    learned_representations: Empty learned_represenations map.
  """

  model_key: str
  model_config: ConfigDict
  time_pooling: str = 'mean'
  channel_pooling: str = 'squeeze'

  # The following are populated during init.
  loaded_model: interface.EmbeddingModel = dataclasses.field(init=False)
  model_callback: eval_lib.EvalModelCallable = dataclasses.field(init=False)
  # We don't use learned_representations with the simple wrapper, but need to
  # provide an empty mapping for the API.
  learned_representations: dict[str, np.ndarray] = dataclasses.field(
      init=True, default_factory=dict
  )

  def __post_init__(self):
    logging.info('Loading separation model...')
    model_class = inference_models.model_class_map()[self.model_key]
    self.loaded_model = model_class(**self.model_config)
    # Set the object's call method as the model_callback.
    self.model_callback = self.__call__

  def __call__(self, inputs: np.ndarray) -> np.ndarray:
    model_outputs = self.loaded_model.batch_embed(inputs)
    # Batched model outputs have shape [B, T, C, D], but we require [B, D].
    return model_outputs.pooled_embeddings(
        self.time_pooling, self.channel_pooling
    )


@dataclasses.dataclass
class HuBERTModelCallback:
  """A model callback implementation for HuBERTModel checkpoints.

  Attributes:
    init_config: TaxonomyModel configuration.
    workdir: path to the model checkpoint.
    embedding_index: index of the embedding vector to retrieve in the list of
      embeddings output by the model.
    model_callback: the fprop function used as part of the model callback,
      created automatically post-initialization.
    learned_representations: mapping from class name to its learned
      representation, created automatically post-initialization and left empty
      (because HuBERT is self-supervised).
  """

  init_config: ConfigDict
  workdir: str
  embedding_index: int
  model_callback: eval_lib.EvalModelCallable = dataclasses.field(init=False)
  learned_representations: dict[str, np.ndarray] = dataclasses.field(
      init=False, default_factory=dict
  )

  def __post_init__(self):
    model_bundle, train_state, _ = hubert.initialize_model(
        workdir=self.workdir, num_train_steps=1, **self.init_config
    )
    train_state = model_bundle.ckpt.restore(train_state)
    variables = {'params': train_state.params, **train_state.model_state}

    @jax.jit
    def fprop(inputs):
      model_outputs = model_bundle.model.apply(
          variables, inputs, train=False, mask_key=None
      )
      return model_outputs.embedding[self.embedding_index].mean(axis=-2)

    self.model_callback = fprop

  def __call__(self, inputs: np.ndarray) -> np.ndarray:
    return np.asarray(self.model_callback(inputs))


@dataclasses.dataclass
class SeparationModelCallback:
  """A model callback implementation for SeparationModel checkpoints.

  Attributes:
    init_config: SeparationModel configuration.
    workdir: path to the model checkpoint.
    model_callback: the fprop function used as part of the model callback,
      created automatically post-initialization.
    learned_representations: mapping from class name to its learned
      representation, created automatically post-initialization. If
      `use_learned_representations` is False, it is left empty, which results in
      the evaluation protocol relying instead on embedded upstream recordings to
      form search queries.
  """

  init_config: ConfigDict
  workdir: str
  model_callback: eval_lib.EvalModelCallable = dataclasses.field(init=False)
  learned_representations: dict[str, np.ndarray] = dataclasses.field(
      init=False, default_factory=dict
  )

  def __post_init__(self):
    model_bundle, train_state = separator.initialize_model(
        workdir=self.workdir, **self.init_config
    )
    train_state = model_bundle.ckpt.restore_or_initialize(train_state)
    variables = {'params': train_state.params, **train_state.model_state}

    @jax.jit
    def fprop(inputs):
      return model_bundle.model.apply(
          variables, inputs, train=False
      ).embedding.mean(axis=-2)

    self.model_callback = fprop

  def __call__(self, inputs: np.ndarray) -> np.ndarray:
    return np.asarray(self.model_callback(inputs))
