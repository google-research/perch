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


from absl import logging
from chirp import export_utils
from chirp.data import pipeline
from chirp.models import metrics
from chirp.models import output
from chirp.models import rank_based_metrics
from chirp.models import separation_model
from chirp.taxonomy import class_utils
from chirp.train import utils
from clu import checkpoint
from clu import metric_writers
from clu import metrics as clu_metrics
from clu import periodic_actions
import flax
import flax.jax_utils as flax_utils
import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from ml_collections import config_dict
import numpy as np
import optax
import tensorflow as tf

EVAL_LOOP_SLEEP_S = 30


def p_log_mse_loss(
    source: jnp.ndarray,
    estimate: jnp.ndarray,
    max_snr: float = 1e6,
    **unused_kwargs,
):
  return lax.pmean(
      jnp.mean(metrics.log_mse_loss(source, estimate, max_snr)),
      axis_name='batch',
  )


def p_log_snr_loss(
    source: jnp.ndarray,
    estimate: jnp.ndarray,
    max_snr: float = 1e6,
    **unused_kwargs,
):
  return lax.pmean(
      jnp.mean(metrics.negative_snr_loss(source, estimate, max_snr)),
      axis_name='batch',
  )


def p_log_sisnr_loss(
    source: jnp.ndarray,
    estimate: jnp.ndarray,
    max_snr: float = 1e6,
    **unused_kwargs,
):
  return lax.pmean(
      jnp.mean(metrics.negative_snr_loss(source, estimate, max_snr=max_snr)),
      axis_name='batch',
  )


def keyed_cross_entropy(
    key: str,
    outputs: separation_model.SeparatorOutput,
    **kwargs,
) -> jnp.ndarray | None:
  """Cross entropy for the specified taxonomic label set."""
  if getattr(outputs, key) is None:
    return 0  # pytype: disable=bad-return-type  # jax-ndarray
  scores = getattr(outputs, key)
  ce = optax.sigmoid_binary_cross_entropy(scores, kwargs[key])
  ce = jnp.mean(ce, axis=-1)
  return ce


def keyed_map(
    key: str, outputs: separation_model.SeparatorOutput, **kwargs
) -> jnp.ndarray | None:
  if getattr(outputs, key) is None:
    return 0  # pytype: disable=bad-return-type  # jax-ndarray
  scores = getattr(outputs, key)
  return metrics.average_precision(scores=scores, labels=kwargs[key])


def make_metrics_collection(prefix: str):
  """Create metrics collection."""
  metrics_dict = {
      'mixit_log_mse': clu_metrics.LastValue.from_fun(p_log_mse_loss),
      'mixit_neg_snr': clu_metrics.LastValue.from_output('mixit_neg_snr'),
  }
  for key in ['label'] + utils.TAXONOMY_KEYS:
    metrics_dict.update({
        f'{key}_xentropy': clu_metrics.Average.from_fun(
            functools.partial(keyed_cross_entropy, key=key)
        ),
        f'{key}_map': clu_metrics.Average.from_fun(
            functools.partial(keyed_map, key=key)
        ),
    })
  metrics_dict['taxo_loss'] = clu_metrics.Average.from_output('taxo_loss')
  metrics_dict = {f'{prefix}_{k}': v for k, v in metrics_dict.items()}
  return clu_metrics.Collection.create(**metrics_dict)


def initialize_model(
    input_shape: tuple[int, ...],
    rng_seed: int,
    learning_rate: float,
    workdir: str,
    model_config: config_dict.ConfigDict,
    target_class_list: str,
):
  """Creates model for training, eval, or inference."""
  # Initialize random number generator
  key = random.PRNGKey(rng_seed)

  # Load model
  model_init_key, key = random.split(key)
  class_lists = class_utils.get_class_lists(target_class_list, True)
  model = separation_model.SeparationModel(
      num_classes={k: v.size for (k, v) in class_lists.items()}, **model_config
  )
  variables = model.init(
      model_init_key, jnp.zeros((1,) + input_shape), train=False
  )
  model_state, params = variables.pop('params')

  # Initialize optimizer
  optimizer = optax.adam(learning_rate=learning_rate)
  opt_state = optimizer.init(params)

  # Load checkpoint
  ckpt = checkpoint.MultihostCheckpoint(workdir)
  train_state = utils.TrainState(
      step=0, params=params, opt_state=opt_state, model_state=model_state
  )
  train_state = ckpt.restore_or_initialize(train_state)
  return (
      utils.ModelBundle(
          model=model,
          key=key,
          ckpt=ckpt,
          optimizer=optimizer,
          class_lists=class_lists,
      ),
      train_state,
  )


def train(
    model_bundle,
    train_state,
    train_dataset,
    num_train_steps: int,
    logdir: str,
    log_every_steps: int,
    checkpoint_every_steps: int,
    loss_max_snr: float,
    classify_bottleneck_weight: float,
    taxonomy_labels_weight: float,
) -> None:
  """Train a model."""
  train_iterator = train_dataset.as_numpy_iterator()
  train_metrics_collection = make_metrics_collection('train__')
  initial_step = int(train_state.step)
  train_state = flax.jax_utils.replicate(train_state)
  # Logging
  writer = metric_writers.create_default_writer(logdir)
  reporter = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer
  )

  @functools.partial(jax.pmap, axis_name='batch')
  def train_step(batch, train_state):
    """Training step for the separation model."""

    def update_step(params, model_state):
      variables = {'params': params, **model_state}
      model_outputs, model_state = model_bundle.model.apply(
          variables,
          batch['audio'],
          train=True,
          mutable=list(model_state.keys()),
      )
      estimate, mixit_matrix = metrics.least_squares_mixit(
          reference=batch['source_audio'],
          estimate=model_outputs.separated_audio,
      )
      model_outputs = model_outputs.time_reduce_logits('MIDPOINT')
      taxo_loss = utils.taxonomy_cross_entropy(
          model_outputs, taxonomy_labels_weight, **batch
      )['loss']
      mixit_neg_snr = p_log_snr_loss(
          batch['source_audio'], estimate, loss_max_snr
      )
      train_metrics = train_metrics_collection.gather_from_model_output(
          outputs=model_outputs,
          separated=model_outputs.separated_audio,
          source=batch['source_audio'],
          estimate=estimate,
          mixit_matrix=mixit_matrix,
          max_snr=loss_max_snr,
          taxo_loss=taxo_loss,
          mixit_neg_snr=mixit_neg_snr,
          **batch,
          **output.logits(model_outputs),
      )
      loss = mixit_neg_snr
      if classify_bottleneck_weight > 0.0:
        loss = mixit_neg_snr + classify_bottleneck_weight * jnp.mean(taxo_loss)
      return loss, (train_metrics, model_state)

    grads, (train_metrics, model_state) = jax.grad(update_step, has_aux=True)(
        train_state.params, train_state.model_state
    )
    grads = jax.lax.pmean(grads, axis_name='batch')
    updates, opt_state = model_bundle.optimizer.update(
        grads, train_state.opt_state
    )
    params = optax.apply_updates(train_state.params, updates)
    train_state = utils.TrainState(
        step=train_state.step + 1,
        params=params,
        opt_state=opt_state,
        model_state=model_state,
    )
    return train_metrics, train_state

  for step in range(initial_step, num_train_steps + 1):
    with jax.profiler.StepTraceAnnotation('train__', step_num=step):
      batch = next(train_iterator)
      train_metrics, train_state = train_step(batch, train_state)

      if step % log_every_steps == 0:
        train_metrics = flax_utils.unreplicate(train_metrics).compute()
        train_metrics = {
            k.replace('___', '/'): v for k, v in train_metrics.items()
        }
        writer.write_scalars(step, utils.flatten_dict(train_metrics))
      reporter(step)
    if step % checkpoint_every_steps == 0:
      with reporter.timed('checkpoint'):
        model_bundle.ckpt.save(flax_utils.unreplicate(train_state))
  writer.close()


def evaluate(
    model_bundle: utils.ModelBundle,
    train_state: utils.TrainState,
    valid_dataset: tf.data.Dataset,
    writer: metric_writers.MetricWriter,
    reporter: periodic_actions.ReportProgress,
    loss_max_snr: float,
    taxonomy_labels_weight: float,
    eval_steps_per_checkpoint: int = -1,
    add_class_wise_metrics: bool = False,
):
  """Run evaluation."""
  base_metrics_collection = make_metrics_collection('valid__')
  valid_metrics_collection = (
      rank_based_metrics.add_rank_based_metrics_to_metrics_collection(
          'valid', base_metrics_collection
      )
  )

  @functools.partial(jax.pmap, axis_name='batch')
  def get_metrics(batch, train_state):
    variables = {'params': train_state.params, **train_state.model_state}
    model_outputs = model_bundle.model.apply(
        variables, batch['audio'], train=False
    )
    model_outputs = model_outputs.time_reduce_logits('MIDPOINT')

    estimate, mixit_matrix = metrics.least_squares_mixit(
        reference=batch['source_audio'], estimate=model_outputs.separated_audio
    )
    taxo_loss = utils.taxonomy_cross_entropy(
        model_outputs, taxonomy_labels_weight, **batch
    )['loss']
    mixit_neg_snr = p_log_snr_loss(
        batch['source_audio'], estimate, loss_max_snr
    )
    return valid_metrics_collection.gather_from_model_output(
        outputs=model_outputs,
        separated=model_outputs.separated_audio,
        source=batch['source_audio'],
        estimate=estimate,
        mixit_matrix=mixit_matrix,
        taxo_loss=taxo_loss,
        mixit_neg_snr=mixit_neg_snr,
        **batch,
        **output.logits(model_outputs),
    )

  with reporter.timed('eval'):
    valid_metrics = valid_metrics_collection.empty()
    for valid_step, batch in enumerate(valid_dataset.as_numpy_iterator()):
      batch = jax.tree_map(np.asarray, batch)
      new_valid_metrics = get_metrics(batch, flax_utils.replicate(train_state))
      valid_metrics = valid_metrics.merge(
          flax_utils.unreplicate(new_valid_metrics)
      )
      if (
          eval_steps_per_checkpoint > 0
          and valid_step >= eval_steps_per_checkpoint
      ):
        break

    # Log validation loss
    valid_metrics = valid_metrics.compute()

  valid_metrics = utils.flatten_dict(
      {k.replace('___', '/'): v for k, v in valid_metrics.items()}
  )
  classwise_metrics = {
      k: v for k, v in valid_metrics.items() if 'individual' in k
  }
  valid_metrics = {
      k: v for k, v in valid_metrics.items() if k not in classwise_metrics
  }

  writer.write_scalars(int(train_state.step), valid_metrics)
  if add_class_wise_metrics:
    writer.write_summaries(int(train_state.step), classwise_metrics)
  writer.flush()


def evaluate_loop(
    model_bundle: utils.ModelBundle,
    train_state: utils.TrainState,
    valid_dataset: tf.data.Dataset,
    workdir: str,
    logdir: str,
    num_train_steps: int,
    eval_steps_per_checkpoint: int,
    loss_max_snr: float,
    taxonomy_labels_weight: float,
    tflite_export: bool = False,
    frame_size: int | None = None,
    eval_sleep_s: int = EVAL_LOOP_SLEEP_S,
    add_class_wise_metrics: bool = False,
):
  """Run evaluation in a loop."""
  writer = metric_writers.create_default_writer(logdir)
  reporter = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer
  )
  # Initialize last_step to -1 so we always run at least one eval.
  last_step = -1
  last_ckpt = ''

  while last_step < num_train_steps:
    ckpt = checkpoint.MultihostCheckpoint(workdir)
    next_ckpt = ckpt.get_latest_checkpoint_to_restore_from()
    if next_ckpt is None or next_ckpt == last_ckpt:
      time.sleep(eval_sleep_s)
      continue
    try:
      train_state = ckpt.restore(train_state, next_ckpt)
      logging.info('Restored checkpoing at step %d', int(train_state.step))
    except tf.errors.NotFoundError:
      logging.warning(
          'Checkpoint %s not found in workdir %s',
          ckpt.latest_checkpoint,
          workdir,
      )
      time.sleep(eval_sleep_s)
      continue

    st = time.time()
    evaluate(
        model_bundle,
        train_state,
        valid_dataset,
        writer,
        reporter,
        loss_max_snr,
        taxonomy_labels_weight,
        eval_steps_per_checkpoint,
        add_class_wise_metrics=add_class_wise_metrics,
    )
    elapsed = time.time() - st
    last_step = int(train_state.step)
    logging.info('Finished eval step %d in %8.2f s', last_step, elapsed)
    if tflite_export:
      st = time.time()
      export_tf(model_bundle, train_state, workdir, frame_size)
      elapsed = time.time() - st
      logging.info('Exported model at step %d in %8.2f s', last_step, elapsed)

    last_ckpt = next_ckpt


def export_tf(
    model_bundle: utils.ModelBundle,
    train_state: utils.TrainState,
    workdir: str,
    frame_size: int,
):
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

  # CAUTION: If the infer_fn signature changes, then the SeparatorTFCallback
  # in the eval benchmark code will also need to be changed.
  def infer_fn(framed_audio_batch, variables):
    flat_inputs = jnp.reshape(
        framed_audio_batch, [framed_audio_batch.shape[0], -1]
    )
    model_outputs = model_bundle.model.apply(
        variables, flat_inputs, train=False
    )
    return (
        model_outputs.separated_audio,
        model_outputs.label,
        model_outputs.embedding,
    )

  converted_model = export_utils.Jax2TfModelWrapper(
      infer_fn, variables, [None, None, frame_size], False
  )
  converted_model.export_converted_model(
      workdir, train_state.step, model_bundle.class_lists
  )


def run(
    mode: str,
    config: config_dict.ConfigDict,
    workdir: str,
    tf_data_service_address: str,
) -> None:
  """Run the experiment."""
  if mode == 'train':
    train_dataset, dataset_info = pipeline.get_dataset(
        is_train=True,
        tf_data_service_address=tf_data_service_address,
        **config.train_dataset_config,
    )
  elif mode == 'eval':
    valid_dataset, dataset_info = pipeline.get_dataset(
        **config.eval_dataset_config
    )
  if dataset_info.features['audio'].sample_rate != config.sample_rate_hz:
    raise ValueError(
        'Dataset sample rate must match config sample rate. To address this, '
        'need to set the sample rate in the config to {}.'.format(
            dataset_info.features['audio'].sample_rate
        )
    )

  model_bundle, train_state = initialize_model(
      workdir=workdir, **config.init_config
  )
  if mode == 'train':
    train_state = model_bundle.ckpt.restore_or_initialize(train_state)
    train(
        model_bundle,
        train_state,
        train_dataset,
        logdir=workdir,
        **config.train_config,
    )
  elif mode == 'eval':
    evaluate_loop(
        model_bundle,
        train_state,
        valid_dataset,
        workdir=workdir,
        logdir=workdir,
        **config.eval_config,
    )
