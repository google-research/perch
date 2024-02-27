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

"""Shared utilities for training scripts."""

import itertools
import os
import time
from typing import Callable, Sequence

from absl import logging
from chirp import path_utils
from chirp.models import output
from chirp.taxonomy import namespace
from chirp.taxonomy import namespace_db
from clu import checkpoint
from clu import metrics as clu_metrics
import flax
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax
import tensorflow as tf

TAXONOMY_KEYS = ['genus', 'family', 'order']


# Note: Inherit from PyTreeNode instead of using the flax.struct.dataclass
# to avoid PyType issues.
# See: https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html
class TrainState(flax.struct.PyTreeNode):
  step: int
  params: flax.core.scope.VariableDict
  opt_state: optax.OptState
  model_state: flax.core.scope.FrozenVariableDict


class OutputHeadMetadata(flax.struct.PyTreeNode):
  """Data describing a classifer output head."""

  key: str
  class_list: namespace.ClassList
  weight: float

  @classmethod
  def from_db(cls, key: str, class_list_name: str, weight: float):
    db = namespace_db.load_db()
    return cls(
        key=key, class_list=db.class_lists[class_list_name], weight=weight
    )

  @classmethod
  def from_mapping(
      cls,
      key: str,
      source_class_list_name: str,
      weight: float,
      mapping_name: str,
      keep_unknown: bool | None = None,
  ):
    db = namespace_db.load_db()
    source_classlist = db.class_lists[source_class_list_name]
    mapping = db.mappings[mapping_name]
    class_list = source_classlist.apply_namespace_mapping(
        mapping, keep_unknown=keep_unknown
    )
    return cls(key=key, class_list=class_list, weight=weight)


class ModelBundle(flax.struct.PyTreeNode):
  model: nn.Module
  key: jnp.ndarray
  ckpt: checkpoint.Checkpoint
  optimizer: optax.GradientTransformation | None = None
  class_lists: dict[str, namespace.ClassList] | None = None
  output_head_metadatas: Sequence[OutputHeadMetadata] | None = None


@flax.struct.dataclass
class MultiAverage(clu_metrics.Average):
  """Computes the average of all values on the last dimension."""

  total: jnp.ndarray
  count: jnp.ndarray

  @classmethod
  def create(cls, n: int):
    return flax.struct.dataclass(
        type('_InlineMultiAverage', (MultiAverage,), {'_n': n})
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
      raise ValueError('expected a vector')
    if mask is None:
      mask = jnp.ones_like(values)
    # Leading dimensions of mask and values must match.
    if mask.shape[0] != values.shape[0]:
      raise ValueError(
          'Argument `mask` must have the same leading dimension as `values`. '
          f'Received mask of dimension {mask.shape} '
          f'and values of dimension {values.shape}.'
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
    return {
        'mean': jnp.sum(self.total) / jnp.sum(self.count),
        'individual': self.total / self.count,
    }


class CollectingMetrics(clu_metrics.Metric):
  """Metrics that must be calculated on collected values.

  To avoid having multiple metrics collect the same values (which could require
  lots of memory) this metric collects all values once, and then applies
  several functions to the collected values to compute metrics.
  """

  @classmethod
  def from_funs(cls, **funs):
    """Construct from a set of functions.

    Args:
      **funs: A mapping from metric names to 2-tuples, where the first element
        is a list of model outputs that need to be collected, and the second
        element is a function which will be applied to the collected model
        outputs in order to calculate the final metric value.

    Returns:
      A metric class that computes metrics using collected values.
    """
    names = list(
        set(
            itertools.chain.from_iterable(metric[0] for metric in funs.values())
        )
    )

    @flax.struct.dataclass
    class FromFuns(clu_metrics.CollectingMetric.from_outputs(names)):
      """Collecting metric which applies functions to collected values."""

      def compute(self):
        """Compute metrics by applying functions to collected values.

        Note that this deviates from the standard `compute` signature, which
        normally returns a scalar or array.

        Returns:
          A dictionary mapping metric names to compute values, which can either
          be scalars/arrays or another dictionary of computed metrics.
        """
        with jax.default_device(jax.local_devices(backend='cpu')[0]):
          values = super().compute()
          return {
              metric_name: metric[1](*(values[name] for name in metric[0]))
              for metric_name, metric in funs.items()
          }

      compute_value = None

    return FromFuns


def flatten(dict_, parent_key='', sep='_'):
  """Recursively flatten dictionaries with string keys.

  Args:
    dict_: The dictionary to flatten.
    parent_key: The name of the parent key.
    sep: The separator used to combine keys.

  Returns:
    A flat dictionary.
  """
  flattened_dict = {}
  for k, v in dict_.items():
    child_key = parent_key + sep + k if parent_key else k
    if isinstance(v, dict):
      flattened_dict |= flatten(v, child_key, sep=sep)
    else:
      flattened_dict[child_key] = v
  return flattened_dict


class NestedCollection(clu_metrics.Collection):
  """Collection that handles metrics which return multiple values."""

  @classmethod
  def create(cls, **metrics):
    # TODO(bartvm): This should be fixed in parent class
    return flax.struct.dataclass(
        type('_InlineCollection', (cls,), {'__annotations__': metrics})
    )

  def compute(self, prefix: str = ''):
    return flatten(super().compute(), parent_key=prefix)

  def compute_values(self, prefix: str = ''):
    return flatten(super().compute_values(), parent_key=prefix)


def write_metrics(writer, step, metrics):
  """Helper function for logging both scalars and arrays."""
  scalars = {k: v for k, v in metrics.items() if v.ndim == 0}
  summaries = {k: v for k, v in metrics.items() if v.ndim != 0}
  writer.write_scalars(step, scalars)
  writer.write_summaries(step, summaries)


def wait_for_next_checkpoint(
    train_state, ckpt, last_ckpt_path, workdir, sleep_s: int = 5
):
  """Wait for the next checkpoint to arrive and load train_state."""
  while True:
    next_ckpt_path = ckpt.get_latest_checkpoint_to_restore_from()
    if next_ckpt_path is None:
      logging.warning('No checkpoint found; sleeping.')
      time.sleep(sleep_s)
      continue
    elif next_ckpt_path == last_ckpt_path:
      logging.warning('No new checkpoint found; sleeping.')
      time.sleep(sleep_s)
      continue
    try:
      new_train_state = ckpt.restore(train_state, next_ckpt_path)
      break
    except tf.errors.NotFoundError:
      logging.warning(
          'Checkpoint %s not found in workdir %s',
          ckpt.latest_checkpoint,
          workdir,
      )
      time.sleep(sleep_s)
      continue
  return new_train_state, next_ckpt_path


def checkpoint_iterator(
    train_state: TrainState,
    ckpt: checkpoint.Checkpoint,
    workdir: str,
    num_train_steps: int,
    sleep_s: int = 5,
):
  """Iterate over checkpoints produced by the train job."""
  last_step = -1
  last_ckpt_path = ''
  elapsed = -1

  st = time.time()
  while last_step < num_train_steps:
    if elapsed is None:
      elapsed = time.time() - st
      logging.info(
          'Finished processing checkpoint %d in %8.2f s', last_step, elapsed
      )

    new_ckpt_path = ckpt.get_latest_checkpoint_to_restore_from()
    if new_ckpt_path is None:
      logging.warning('No checkpoint found; sleeping.')
      time.sleep(sleep_s)
      continue
    elif new_ckpt_path == last_ckpt_path:
      logging.warning('No new checkpoint found; sleeping.')
      time.sleep(sleep_s)
      continue
    try:
      new_train_state = ckpt.restore(train_state, new_ckpt_path)
    except tf.errors.NotFoundError:
      logging.warning(
          'Checkpoint %s not found in workdir %s',
          ckpt.latest_checkpoint,
          workdir,
      )
      time.sleep(sleep_s)
      continue
    except Exception as error:
      logging.warning(
          'Unknown exception %s not found in workdir %s',
          ckpt.latest_checkpoint,
          workdir,
      )
      logging.error(error)
      time.sleep(sleep_s)
      continue
    last_ckpt_path = new_ckpt_path
    train_state = new_train_state
    last_step = int(train_state.step)
    elapsed = None
    st = time.time()
    logging.info('Loaded checkpoint at step %d', int(train_state.step))
    yield train_state


def output_head_loss(
    outputs: dict[str, jnp.ndarray],
    output_head_metadatas: Sequence[OutputHeadMetadata],
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    **kwargs,
) -> dict[str, jnp.ndarray]:
  """Compute losses from model outputs and output head specifications."""
  total_loss = jnp.array(0.0)
  losses = {}
  for md in output_head_metadatas:
    md_loss = loss_fn(outputs[md.key], kwargs[md.key])
    losses[f'{md.key}_loss'] = md_loss
    total_loss += md.weight * jnp.mean(md_loss, axis=-1)
  losses['loss'] = total_loss
  return losses


def taxonomy_loss(
    outputs: output.TaxonomicOutput,
    taxonomy_loss_weight: float,
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    **kwargs,
) -> jnp.ndarray:
  """Computes the mean loss across taxonomic labels."""
  losses = {'label_loss': loss_fn(getattr(outputs, 'label'), kwargs['label'])}
  losses['loss'] = jnp.mean(losses['label_loss'], axis=-1)
  if taxonomy_loss_weight != 0:
    losses.update(
        {
            f'{key}_loss': loss_fn(getattr(outputs, key), kwargs[key])
            for key in TAXONOMY_KEYS
            if key in kwargs
        }
    )
    losses['loss'] = losses['loss'] + sum(
        taxonomy_loss_weight * jnp.mean(losses[f'{key}_loss'], axis=-1)
        for key in TAXONOMY_KEYS
    )
  return losses  # pytype: disable=bad-return-type  # jax-ndarray
