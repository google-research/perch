# coding=utf-8
# Copyright 2023 The BIRB Authors.
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
from birb.eval import eval_lib
from birb.taxonomy import namespace
from birb.taxonomy import namespace_db
from birb.train import classifier
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
      return outputs
    else:
      # If not, run the remainder of the batch on each host
      batch = inputs[n * (m // n) :]
      remainder = model_callable_jit(batch)
      return jnp.concatenate([outputs, remainder])

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
      class_list = (
          namespace_db.load_db()
          .class_lists[self.init_config.target_class_list]
          .classes
      )
      head_index = list(model_bundle.model.num_classes.keys()).index('label')
      output_weights = train_state.params[f'Dense_{head_index}']['kernel'].T
      self.learned_representations.update(
          {
              n: w
              for n, w in zip(class_list, output_weights)
              if n not in self.learned_representation_blocklist
          }
      )

  def __call__(self, inputs: np.ndarray) -> np.ndarray:
    return np.asarray(self.model_callback(inputs))
