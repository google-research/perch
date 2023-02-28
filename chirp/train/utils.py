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

"""Shared utilities for training scripts."""

from chirp.models import output
from chirp.taxonomy import namespace
from clu import checkpoint
from clu import metrics as clu_metrics
import flax
from flax import linen as nn
from jax import numpy as jnp
import numpy as np
import optax


TAXONOMY_KEYS = ["genus", "family", "order"]


@flax.struct.dataclass
class TrainState:
  step: int
  params: flax.core.scope.VariableDict
  opt_state: optax.OptState
  model_state: flax.core.scope.FrozenVariableDict


@flax.struct.dataclass
class ModelBundle:
  model: nn.Module
  optimizer: optax.GradientTransformation
  key: jnp.ndarray
  ckpt: checkpoint.Checkpoint
  class_lists: dict[str, namespace.ClassList] | None = None


@flax.struct.dataclass
class MultiAverage(clu_metrics.Average):
  """Computes the average of all values on the last dimension."""

  total: jnp.array
  count: jnp.array

  @classmethod
  def create(cls, n: int):
    return flax.struct.dataclass(
        type("_InlineMultiAverage", (MultiAverage,), {"_n": n})
    )

  @classmethod
  def empty(cls) -> clu_metrics.Metric:
    # pytype: disable=attribute-error
    return cls(
        total=jnp.zeros(cls._n, jnp.float32), count=jnp.zeros(cls._n, jnp.int32)
    )
    # pytype: enable=attribute-error

  @classmethod
  def from_model_output(
      cls, values: jnp.ndarray, mask: jnp.ndarray | None = None, **_
  ) -> clu_metrics.Metric:
    if values.ndim == 0:
      raise ValueError("expected a vector")
    if mask is None:
      mask = jnp.ones_like(values)
    # Leading dimensions of mask and values must match.
    if mask.shape[0] != values.shape[0]:
      raise ValueError(
          "Argument `mask` must have the same leading dimension as `values`. "
          f"Received mask of dimension {mask.shape} "
          f"and values of dimension {values.shape}."
      )
    # Broadcast mask to the same number of dimensions as values.
    if mask.ndim < values.ndim:
      mask = jnp.expand_dims(
          mask, axis=tuple(np.arange(mask.ndim, values.ndim))
      )
    mask = mask.astype(bool)
    axes = tuple(np.arange(values.ndim - 1))
    return cls(
        total=jnp.where(mask, values, jnp.zeros_like(values)).sum(axis=axes),
        count=jnp.where(
            mask,
            jnp.ones_like(values, dtype=jnp.int32),
            jnp.zeros_like(values, dtype=jnp.int32),
        ).sum(axis=axes),
    )

  def compute(self):
    averages = self.total / self.count
    return {
        "mean": jnp.sum(self.total) / jnp.sum(self.count),
        **{str(i): averages[i] for i in range(jnp.size(averages))},
    }


def flatten_dict(
    metrics: dict[str, jnp.ndarray | dict[str, jnp.ndarray]]
) -> dict[str, jnp.ndarray]:
  """Flatten a metrics dictionary.

  The `MultiAverage` metric actually returns a dictionary instead of a scalar.
  After calling `compute()` the resulting values must be flattened using this
  function becfore being passed to `write_scalars`.

  Args:
    metrics: A dictionary with metrics where the values are either scalars or
      dictionaries that map strings to scalars.

  Returns:
    A flat dictionary where each key maps to a scalar.
  """
  flattened_dict = {}
  for k, v in metrics.items():
    if isinstance(v, dict):
      flattened_dict.update({f"{k}_{subk}": subv for subk, subv in v.items()})
    else:
      flattened_dict[k] = v
  return flattened_dict


def taxonomy_cross_entropy(
    outputs: output.TaxonomicOutput,
    taxonomy_loss_weight: float,
    **kwargs,
) -> jnp.ndarray:
  """Computes mean cross entropy across taxonomic labels."""
  xentropy = {
      f"{key}_xentropy": optax.sigmoid_binary_cross_entropy(
          getattr(outputs, key), kwargs[key]
      )
      for key in ["label"] + TAXONOMY_KEYS
      if key in kwargs
  }
  xentropy["loss"] = jnp.mean(xentropy["label_xentropy"], axis=-1)
  if taxonomy_loss_weight != 0:
    xentropy["loss"] = xentropy["loss"] + sum(
        taxonomy_loss_weight * jnp.mean(xentropy[f"{key}_xentropy"], axis=-1)
        for key in TAXONOMY_KEYS
    )
  return xentropy  # pytype: disable=bad-return-type  # jax-ndarray
