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

"""Training loop for separation models."""

import functools
import time
from typing import Dict, Optional

from absl import logging
from chirp import export_utils
from chirp.models import cmap
from chirp.models import metrics
from chirp.models import separation_model
from chirp.taxonomy import class_utils
from chirp.taxonomy import namespace
from clu import checkpoint
from clu import metric_writers
from clu import metrics as clu_metrics
from clu import periodic_actions
import flax
from flax import linen as nn
import flax.jax_utils as flax_utils
import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from ml_collections import config_dict
import optax
import tensorflow as tf

EVAL_LOOP_SLEEP_S = 30


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
  class_lists: Dict[str, namespace.ClassList]


def p_log_mse_loss(source: jnp.ndarray,
                   estimate: jnp.ndarray,
                   max_snr: float = 1e6,
                   **unused_kwargs):
  return lax.pmean(
      jnp.mean(metrics.log_mse_loss(source, estimate, max_snr)),
      axis_name='batch')


def p_log_snr_loss(source: jnp.ndarray,
                   estimate: jnp.ndarray,
                   max_snr: float = 1e6,
                   **unused_kwargs):
  return lax.pmean(
      jnp.mean(metrics.negative_snr_loss(source, estimate, max_snr)),
      axis_name='batch')


def p_log_sisnr_loss(source: jnp.ndarray,
                     estimate: jnp.ndarray,
                     max_snr: float = 1e6,
                     **unused_kwargs):
  return lax.pmean(
      jnp.mean(metrics.negative_snr_loss(source, estimate, max_snr=max_snr)),
      axis_name='batch')


@flax.struct.dataclass
class ValidationMetrics(clu_metrics.Collection):
  valid_loss: clu_metrics.Average.from_fun(p_log_snr_loss)
  valid_mixit_log_mse: clu_metrics.Average.from_fun(p_log_mse_loss)
  valid_mixit_neg_snr: clu_metrics.Average.from_fun(p_log_snr_loss)


@flax.struct.dataclass
class TrainingMetrics(clu_metrics.Collection):
  train_loss: clu_metrics.LastValue.from_fun(p_log_snr_loss)
  train_mixit_log_mse: clu_metrics.LastValue.from_fun(p_log_mse_loss)
  train_mixit_neg_snr: clu_metrics.LastValue.from_fun(p_log_snr_loss)


def taxonomy_cross_entropy(outputs: separation_model.ModelOutputs,
                           label: jnp.ndarray,
                           genus: jnp.ndarray,
                           family: jnp.ndarray,
                           order: jnp.ndarray,
                           taxonomy_labels_weight: float = 0.001,
                           **unused_kwargs) -> jnp.ndarray:
  """Computes mean cross entropy across taxonomic labels."""
  # Note that the classification outputs are reduced to shape [B, D] from
  # [B, T, D] prior to the loss computation.
  if outputs.label is None:
    return 0
  mean = jnp.mean(
      optax.sigmoid_binary_cross_entropy(outputs.label, label), axis=-1)
  mean += taxonomy_labels_weight * jnp.mean(
      optax.sigmoid_binary_cross_entropy(outputs.genus, genus), axis=-1)
  mean += taxonomy_labels_weight * jnp.mean(
      optax.sigmoid_binary_cross_entropy(outputs.family, family), axis=-1)
  mean += taxonomy_labels_weight * jnp.mean(
      optax.sigmoid_binary_cross_entropy(outputs.order, order), axis=-1)
  return mean


def keyed_cross_entropy(key: str, outputs: separation_model.ModelOutputs,
                        **kwargs) -> Optional[jnp.ndarray]:
  """Cross entropy for the specified taxonomic label set."""
  if getattr(outputs, key) is None:
    return 0
  scores = getattr(outputs, key)
  mean = jnp.mean(
      optax.sigmoid_binary_cross_entropy(scores, kwargs[key]), axis=-1)
  return mean


def keyed_map(key: str, outputs: separation_model.ModelOutputs,
              **kwargs) -> Optional[jnp.ndarray]:
  if getattr(outputs, key) is None:
    return 0
  scores = getattr(outputs, key)
  return metrics.average_precision(scores=scores, labels=kwargs[key])


def make_metrics_collection(prefix: str):
  """Create metrics collection."""
  metrics_dict = {
      'mixit_log_mse': clu_metrics.LastValue.from_fun(p_log_mse_loss),
      'mixit_neg_snr': clu_metrics.LastValue.from_fun(p_log_snr_loss),
  }
  taxo_keys = ['label', 'genus', 'family', 'order']
  for key in taxo_keys:
    metrics_dict.update({
        key + '_xentropy':
            clu_metrics.Average.from_fun(
                functools.partial(keyed_cross_entropy, key=key)),
        key + '_map':
            clu_metrics.Average.from_fun(functools.partial(keyed_map, key=key)),
    })
  metrics_dict['taxo_loss'] = clu_metrics.Average.from_fun(
      taxonomy_cross_entropy)
  metrics_dict = {prefix + k: v for k, v in metrics_dict.items()}
  return clu_metrics.Collection.create(**metrics_dict)


def initialize_model(input_size: int, rng_seed: int, learning_rate: float,
                     workdir: str, model_config: config_dict.ConfigDict,
                     target_class_list: str):
  """Creates model for training, eval, or inference."""
  # Initialize random number generator
  key = random.PRNGKey(rng_seed)

  # Load model
  model_init_key, key = random.split(key)
  class_lists = class_utils.get_class_lists(target_class_list, True)
  model = separation_model.SeparationModel(
      num_classes={k: v.size for (k, v) in class_lists.items()}, **model_config)
  variables = model.init(
      model_init_key, jnp.zeros((1, input_size)), train=False)
  model_state, params = variables.pop('params')

  # Initialize optimizer
  optimizer = optax.adam(learning_rate=learning_rate)
  opt_state = optimizer.init(params)

  # Load checkpoint
  ckpt = checkpoint.MultihostCheckpoint(workdir)
  train_state = TrainState(
      step=0, params=params, opt_state=opt_state, model_state=model_state)
  train_state = ckpt.restore_or_initialize(train_state)
  return ModelBundle(model, optimizer, key, ckpt, class_lists), train_state


def train(model_bundle, train_state, train_dataset, num_train_steps: int,
          logdir: str, log_every_steps: int, checkpoint_every_steps: int,
          loss_max_snr: float, classify_bottleneck_weight: float,
          taxonomy_labels_weight: float) -> None:
  """Train a model."""
  train_iterator = train_dataset.as_numpy_iterator()
  train_metrics_collection = make_metrics_collection('train___')
  initial_step = int(train_state.step)
  train_state = flax.jax_utils.replicate(train_state)
  # Logging
  writer = metric_writers.create_default_writer(logdir)
  reporter = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer)

  @functools.partial(jax.pmap, axis_name='batch')
  def train_step(batch, train_state):
    """Training step for the separation model."""

    def update_step(params, model_state):
      variables = {'params': params, **model_state}
      model_outputs, model_state = model_bundle.model.apply(
          variables,
          batch['audio'],
          train=True,
          mutable=list(model_state.keys()))
      estimate, mixit_matrix = metrics.least_squares_mixit(
          reference=batch['source_audio'],
          estimate=model_outputs.separated_audio)
      if 'label' in batch:
        labels = {
            'label': batch['label'],
            'genus': batch['genus'],
            'family': batch['family'],
            'order': batch['order']
        }
      else:
        labels = {}
      model_outputs = model_outputs.time_reduce_logits('MIDPOINT')
      train_metrics = train_metrics_collection.gather_from_model_output(
          outputs=model_outputs,
          separated=model_outputs.separated_audio,
          source=batch['source_audio'],
          estimate=estimate,
          mixit_matrix=mixit_matrix,
          max_snr=loss_max_snr,
          taxonomy_labels_weight=taxonomy_labels_weight,
          **labels).compute()
      loss = train_metrics['train___mixit_neg_snr']
      if classify_bottleneck_weight > 0.0:
        loss += classify_bottleneck_weight * train_metrics['train___taxo_loss']
      return loss, (train_metrics, model_state)

    (_, (train_metrics, model_state)), grads = jax.value_and_grad(
        update_step, has_aux=True)(train_state.params, train_state.model_state)
    grads = jax.lax.pmean(grads, axis_name='batch')
    updates, opt_state = model_bundle.optimizer.update(grads,
                                                       train_state.opt_state)
    params = optax.apply_updates(train_state.params, updates)
    train_state = TrainState(
        step=train_state.step + 1,
        params=params,
        opt_state=opt_state,
        model_state=model_state)
    return train_metrics, train_state

  for step in range(initial_step, num_train_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      batch = next(train_iterator)
      train_metrics, train_state = train_step(batch, train_state)
      train_metrics = flax_utils.unreplicate(train_metrics)

      if step % log_every_steps == 0:
        writer.write_scalars(step, train_metrics)
      reporter(step)
    if step % checkpoint_every_steps == 0:
      with reporter.timed('checkpoint'):
        model_bundle.ckpt.save(flax_utils.unreplicate(train_state))
  writer.close()


def evaluate(model_bundle: ModelBundle,
             train_state: TrainState,
             valid_dataset: tf.data.Dataset,
             writer: metric_writers.MetricWriter,
             reporter: periodic_actions.ReportProgress,
             max_eval_steps: int = -1):
  """Run evaluation."""
  step = train_state.step
  valid_metrics = make_metrics_collection('valid___')

  @functools.partial(jax.pmap, axis_name='batch')
  def evaluate_step(valid_metrics, batch, train_state):
    variables = {'params': train_state.params, **train_state.model_state}
    model_outputs = model_bundle.model.apply(
        variables, batch['audio'], train=False)
    estimate, mixit_matrix = metrics.least_squares_mixit(
        reference=batch['source_audio'], estimate=model_outputs.separated_audio)
    if 'label' in batch:
      labels = {
          'label': batch['label'],
          'genus': batch['genus'],
          'family': batch['family'],
          'order': batch['order']
      }
    else:
      labels = {}
    model_outputs = model_outputs.time_reduce_logits('MIDPOINT')
    return model_outputs, valid_metrics.merge(
        ValidationMetrics.gather_from_model_output(
            outputs=model_outputs,
            separated=model_outputs.separated_audio,
            source=batch['source_audio'],
            estimate=estimate,
            mixit_matrix=mixit_matrix,
            axis_name='batch',
            **labels))

  with reporter.timed('eval'):
    valid_metrics = flax.jax_utils.replicate(ValidationMetrics.empty())
    cmap_metrics = cmap.make_cmap_metrics_dict(
        ('label', 'genus', 'family', 'order'))
    for valid_step, batch in enumerate(valid_dataset.as_numpy_iterator()):
      if max_eval_steps > 0 and valid_step >= max_eval_steps:
        break
      model_outputs, valid_metrics = evaluate_step(
          valid_metrics, batch, flax_utils.replicate(train_state))
      cmap_metrics = cmap.update_cmap_metrics_dict(cmap_metrics, model_outputs,
                                                   batch)

    # Log validation loss
    valid_metrics = flax_utils.unreplicate(valid_metrics)
    valid_metrics = valid_metrics.compute()

  valid_metrics = {k.replace('___', '/'): v for k, v in valid_metrics.items()}
  cmap_metrics = flax_utils.unreplicate(cmap_metrics)
  for key in cmap_metrics:
    valid_metrics[f'valid/{key}_cmap'] = cmap_metrics[key].compute()
  writer.write_scalars(int(step), valid_metrics)
  writer.flush()


def evaluate_loop(model_bundle: ModelBundle,
                  train_state: TrainState,
                  valid_dataset: tf.data.Dataset,
                  workdir: str,
                  logdir: str,
                  num_train_steps: int,
                  eval_steps_per_checkpoint: int,
                  tflite_export: bool = False,
                  frame_size: Optional[int] = None,
                  eval_sleep_s: int = EVAL_LOOP_SLEEP_S):
  """Run evaluation in a loop."""
  writer = metric_writers.create_default_writer(logdir)
  reporter = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer)
  # Initialize last_step to -1 so we always run at least one eval.
  last_step = -1
  last_ckpt = ''

  while last_step < num_train_steps:
    ckpt = checkpoint.MultihostCheckpoint(workdir)
    if ckpt.latest_checkpoint == last_ckpt:
      time.sleep(eval_sleep_s)
      continue
    try:
      train_state = ckpt.restore_or_initialize(train_state)
    except tf.errors.NotFoundError:
      logging.warning('Checkpoint %s not found in workdir %s',
                      ckpt.latest_checkpoint, workdir)
      time.sleep(eval_sleep_s)
      continue

    evaluate(model_bundle, train_state, valid_dataset, writer, reporter,
             eval_steps_per_checkpoint)
    if tflite_export:
      export_tf(model_bundle, train_state, workdir, frame_size)
    last_step = int(train_state.step)
    last_ckpt = ckpt.latest_checkpoint


def export_tf(model_bundle: ModelBundle, train_state: TrainState, workdir: str,
              frame_size: int):
  """Write a TFLite flatbuffer.

  Args:
    model_bundle: The model bundle.
    train_state: The train state.
    workdir: Where to place the exported model.
    frame_size: Frame size for input audio. The exported model will take inputs
      with shape [B, T//frame_size, frame_size]. This ensures that the time
      dimension is divisible by the product of all model strides, which allows
      us to set a polymorphic time dimension. Thus, the frame_size must be
      divisible by the product of all strides in the model.
  """
  variables = {'params': train_state.params, **train_state.model_state}

  def infer_fn(framed_audio_batch, variables):
    flat_inputs = jnp.reshape(framed_audio_batch,
                              [framed_audio_batch.shape[0], -1])
    model_outputs = model_bundle.model.apply(
        variables, flat_inputs, train=False)
    return (model_outputs.separated_audio, model_outputs.label,
            model_outputs.embedding)

  converted_model = export_utils.Jax2TfModelWrapper(infer_fn, variables,
                                                    [None, None, frame_size],
                                                    False)
  converted_model.export_converted_model(workdir, train_state.step,
                                         model_bundle.class_lists)
