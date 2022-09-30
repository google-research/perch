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

"""TrainState utility class."""

from typing import Dict

from clu import checkpoint
import flax
import flax.jax_utils as flax_utils
import jax
from jax import numpy as jnp
import optax


@flax.struct.dataclass
class TrainState:
  """TrainState for Chirp models.

  Attributes:
    step: Training step.
    params: Tree of model params.
    opt_state: Optimizer state.
    model_state: Non-param model state, eg, variables not updated by gradient
      descent.
    rngs: Dictionary of namesd RNGs.
  """
  step: int
  params: flax.core.scope.VariableDict
  opt_state: optax.OptState
  model_state: flax.core.scope.FrozenVariableDict
  rngs: Dict[str, jnp.ndarray]

  def replicate(self):
    """Creates a replicated TrainState instance."""
    # Create a new rng for each replicate.
    new_rngs = {}
    for rng_name, rng_key in self.rngs.items():
      split_keys = jax.random.split(rng_key, num=jax.local_device_count() + 1)
      new_rngs[rng_name] = split_keys[:-1]
      rng_key = split_keys[-1]

    return TrainState(
        flax_utils.replicate(self.step),
        flax_utils.replicate(self.params),
        flax_utils.replicate(self.opt_state),
        flax_utils.replicate(self.model_state),
        new_rngs,
    )

  def unreplicate(self):
    """Creates an unreplicated TrainState instance.

    Note: When used for checkpointing, care must be taken to ensure
    replicability. See save_checkpoint_replicated for a good solution.

    Returns:
      Unreplicated TrainState instance.
    """
    return TrainState(
        flax_utils.unreplicate(self.step),
        flax_utils.unreplicate(self.params),
        flax_utils.unreplicate(self.opt_state),
        flax_utils.unreplicate(self.model_state),
        {rng_name: keys[0] for rng_name, keys in self.rngs.items()},
    )

  def save_checkpoint_replicated(self, ckpt: checkpoint.MultihostCheckpoint):
    """Save a checkpoint and update replicated train_state.

    When saving a checkpoint, we save unreplicated params, and unreplicated
    RNG keys. When restoring, we will invariably replicate the RNG keys.
    Then for consistency, we need to update the train_state when saving a
    checkpoint by unreplicating, saving, then re-replicating, which will
    split the RNGs as we would for a freshly-restored checkpoint.
    Note that this strategy only works so long as the number of replicas is the
    same on save and reload.

    Args:
      ckpt: CLU MultihostCheckpoint object used to save the train_state.

    Returns:
      Replicated TrainState.
    """
    unreplicated = self.unreplicate()
    ckpt.save(unreplicated)
    return unreplicated.replicate()

  def increment(self, params: flax.core.scope.VariableDict,
                opt_state: optax.OptState,
                model_state: flax.core.scope.FrozenVariableDict):
    """Create a new TrainState with incremented step and new rng keys."""
    new_rngs = {}
    for rng_name, rng_key in self.rngs.items():
      new_key, _ = jax.random.split(rng_key)
      new_rngs[rng_name] = new_key

    return TrainState(self.step + 1, params, opt_state, model_state, new_rngs)

  @classmethod
  def make_rngs(cls, rng_seed, rng_names):
    key = jax.random.PRNGKey(rng_seed)
    rngs = {}
    for k in rng_names:
      new_key, key = jax.random.split(key)
      rngs[k] = new_key
    return rngs
