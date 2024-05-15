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

"""Training loop for separation models."""

import functools
import io
from typing import Callable, Dict

from absl import logging
from chirp import export_utils
from chirp.data import utils as data_utils
from chirp.models import metrics
from chirp.models import output
from chirp.models import separation_model
from chirp.taxonomy import class_utils
from chirp.train import train_utils
from clu import checkpoint
from clu import metric_writers
from clu import metrics as clu_metrics
from clu import periodic_actions
import flax
import flax.jax_utils as flax_utils
import imageio as iio
import jax
from jax import numpy as jnp
from jax import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
from ml_collections import config_dict
import numpy as np
import optax
import tensorflow as tf


EVAL_LOOP_SLEEP_S = 30


def p_log_snr_loss(
    source: jnp.ndarray,
    estimate: jnp.ndarray,
    max_snr: float = 1e6,
    **unused_kwargs,
):
  return jnp.mean(metrics.negative_snr_loss(source, estimate, max_snr))


TRAIN_METRICS = {
    'loss': clu_metrics.Average.from_output('loss'),
    'taxo_loss': clu_metrics.Average.from_output('taxo_loss'),
    'mixit_neg_snr': clu_metrics.Average.from_output('mixit_neg_snr'),
}

EVAL_METRICS = {
    'rank_metrics': train_utils.CollectingMetrics.from_funs(
        **{
            'label_cmap': (('label_logits', 'label'), metrics.cmap),
            'genus_cmap': (('genus_logits', 'genus'), metrics.cmap),
            'family_cmap': (('family_logits', 'family'), metrics.cmap),
            'order_cmap': (('order_logits', 'order'), metrics.cmap),
            'label_roc_auc': (('label_logits', 'label'), metrics.roc_auc),
            'genus_roc_auc': (('genus_logits', 'genus'), metrics.roc_auc),
            'family_roc_auc': (('family_logits', 'family'), metrics.roc_auc),
            'order_roc_auc': (('order_logits', 'order'), metrics.roc_auc),
        }
    )
}


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
      num_classes={k: len(v.classes) for (k, v) in class_lists.items()},
      **model_config,
  )
  variables = model.init(
      model_init_key, jnp.zeros((1,) + input_shape), train=False
  )
  model_state, params = flax.core.pop(variables, 'params')

  # Initialize optimizer
  optimizer = optax.adam(learning_rate=learning_rate)
  opt_state = optimizer.init(params)

  # Load checkpoint
  ckpt = checkpoint.MultihostCheckpoint(workdir)
  train_state = train_utils.TrainState(
      step=0, params=params, opt_state=opt_state, model_state=model_state
  )
  train_state = ckpt.restore_or_initialize(train_state)
  return (
      train_utils.ModelBundle(
          model=model,
          key=key,
          ckpt=ckpt,
          optimizer=optimizer,
          class_lists=class_lists,
      ),
      train_state,
  )


def fig_image(fig):
  """Returns an image summary from a matplotlib figure."""
  buffer = io.BytesIO()
  fig.savefig(buffer, format='png', bbox_inches='tight')
  img = iio.imread(buffer.getvalue(), format='png')
  plt.close(fig)
  return img


def force_numpy(arr):
  """Ensures that arr is a numpy array."""
  if isinstance(arr, np.ndarray):
    return arr

  if hasattr(arr, 'numpy'):
    # Eager mode.
    return arr.numpy()
  else:
    return tf.make_ndarray(arr)


def _audio_and_spectrogram_summaries(
    writer: metric_writers.MetricWriter,
    step: int,
    title: str,
    batch: Dict[str, jnp.ndarray],
    separated_audio: jnp.ndarray,
    sample_rate: int = 16000,
    max_outputs: int = 5,
):
  """Makes audio and spectrogram summaries for MixIT models."""

  def _plot_spec(data, ax, title_, do_db_scaling=True):
    if data.ndim == 2:
      spec = data
    else:
      spec = np.abs(librosa.stft(np.array(data)))

    if do_db_scaling:
      spec = librosa.amplitude_to_db(spec, ref=np.max)

    librosa.display.specshow(
        spec,
        y_axis='mel',
        sr=sample_rate,
        x_axis='time',
        ax=ax,
    )
    ax.set(title=title_)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.label_outer()

  # For waveforms, we expect:
  # separated_audio.shape = (tpus, batch // tpus, n_src, time)
  # batch['source_audio'].shape = (tpus, batch // tpus, n_mix, time)
  # batch['audio'].shape = (tpus, batch // tpus, time)

  mixes_of_mixes = np.reshape(batch['audio'], (-1, *batch['audio'].shape[-1:]))
  mixes = np.reshape(
      batch['source_audio'], (-1, *batch['source_audio'].shape[2:])
  )
  separated_audio = np.reshape(
      separated_audio, (-1, *separated_audio.shape[-2:])
  )

  batch_size, n_src, *_ = separated_audio.shape
  n_mixes = mixes.shape[1]
  img_size = 4
  n_rows = max(n_src, n_mixes + 1)

  for i in range(min(batch_size, max_outputs)):
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(2 * img_size, img_size * n_rows)
    )

    # Make summary for Mixture of Mixtures (MoM)
    mom = mixes_of_mixes[i]
    mom_title = 'MoM'
    _plot_spec(mom, axes[0, 0], mom_title)
    writer.write_audios(
        step, {f'{title}/mom{i}': mom[None, :, None]}, sample_rate=sample_rate
    )

    # Make summaries for mixes
    for m in range(n_mixes):
      mix = mixes[i, m, ...]
      mix_title = f'Mix {m+1}'
      _plot_spec(mix, axes[m + 1, 0], mix_title)

      writer.write_audios(
          step,
          {f'{title}/mix{i}_{m}': mix[None, :, None]},
          sample_rate=sample_rate,
      )

    # Make summaries for estimated sources
    for s in range(n_src):
      src = separated_audio[i, s, ...]
      src_title = f'Est. Source {s+1}'
      _plot_spec(src, axes[s, 1], src_title)

      writer.write_audios(
          step,
          {f'{title}/est{i}_src{s}': src[None, :, None]},
          sample_rate=sample_rate,
      )

    # Clean up image and turn into a summary
    for ax in axes.flatten():
      ax.label_outer()
      if not ax.has_data():
        ax.set_visible(False)

    plt.tight_layout()
    img = fig_image(fig)
    writer.write_images(step, {f'{title}/spectrograms_{i}': img[None, ...]})


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
    loss_fn: Callable[
        [jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = optax.sigmoid_binary_cross_entropy,
) -> None:
  """Train a model."""
  if train_dataset is None:
    raise ValueError('train_dataset is None')
  train_iterator = train_dataset.as_numpy_iterator()
  train_metrics_collection = train_utils.NestedCollection.create(
      **TRAIN_METRICS
  )
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
      estimate, _ = metrics.least_squares_mixit(
          reference=batch['source_audio'],
          estimate=model_outputs.separated_audio,
      )
      model_outputs = model_outputs.time_reduce_logits('MIDPOINT')
      taxo_loss = train_utils.taxonomy_loss(
          outputs=model_outputs,
          taxonomy_loss_weight=taxonomy_labels_weight,
          loss_fn=loss_fn,
          **batch,
      )['loss']
      mixit_neg_snr = p_log_snr_loss(
          batch['source_audio'], estimate, loss_max_snr
      )

      loss = mixit_neg_snr
      if classify_bottleneck_weight > 0.0:
        loss = mixit_neg_snr + classify_bottleneck_weight * jnp.mean(taxo_loss)
      train_metrics = train_metrics_collection.gather_from_model_output(
          taxo_loss=taxo_loss,
          mixit_neg_snr=mixit_neg_snr,
          loss=loss,
          **batch,
          **output.logits(model_outputs),
      )
      return loss, (train_metrics, model_state)

    grads, (train_metrics, model_state) = jax.grad(update_step, has_aux=True)(
        train_state.params, train_state.model_state
    )
    grads = jax.lax.pmean(grads, axis_name='batch')
    if model_bundle.optimizer is None:
      raise ValueError('model_bundle.optimizer is None')
    updates, opt_state = model_bundle.optimizer.update(
        grads, train_state.opt_state
    )
    params = optax.apply_updates(train_state.params, updates)
    train_state = train_utils.TrainState(
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
        train_metrics = flax_utils.unreplicate(train_metrics).compute(
            prefix='train'
        )
        train_utils.write_metrics(writer, step, train_metrics)
      reporter(step)
    if step % checkpoint_every_steps == 0:
      with reporter.timed('checkpoint'):
        model_bundle.ckpt.save(flax_utils.unreplicate(train_state))
  writer.close()


def evaluate(
    model_bundle: train_utils.ModelBundle,
    train_state: train_utils.TrainState,
    valid_dataset: tf.data.Dataset,
    workdir: str,
    num_train_steps: int,
    loss_max_snr: float,
    taxonomy_labels_weight: float,
    loss_fn: Callable[
        [jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = optax.sigmoid_binary_cross_entropy,
    eval_sleep_s: int = EVAL_LOOP_SLEEP_S,
    eval_steps_per_checkpoint: int = -1,
    sample_rate_hz: int = 32_000,  # TODO(emanilow): pipe through sample rates.
):
  """Run evaluation."""
  train_metrics = TRAIN_METRICS.copy()
  del train_metrics['loss']
  valid_metrics_collection = train_utils.NestedCollection.create(
      **(train_metrics | EVAL_METRICS)
  )

  @functools.partial(jax.pmap, axis_name='batch')
  def get_metrics(batch, train_state):
    variables = {'params': train_state.params, **train_state.model_state}
    model_outputs = model_bundle.model.apply(
        variables, batch['audio'], train=False
    )
    model_outputs = model_outputs.time_reduce_logits('MIDPOINT')

    estimate, _ = metrics.least_squares_mixit(
        reference=batch['source_audio'], estimate=model_outputs.separated_audio
    )
    taxo_loss = train_utils.taxonomy_loss(
        outputs=model_outputs,
        taxonomy_loss_weight=taxonomy_labels_weight,
        loss_fn=loss_fn,
        **batch,
    )['loss']
    mixit_neg_snr = p_log_snr_loss(
        batch['source_audio'], estimate, loss_max_snr
    )
    return model_outputs, valid_metrics_collection.gather_from_model_output(
        taxo_loss=taxo_loss,
        mixit_neg_snr=mixit_neg_snr,
        **batch,
        **output.logits(model_outputs),
    )

  writer = metric_writers.create_default_writer(workdir, asynchronous=False)
  reporter = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer
  )
  for train_state in train_utils.checkpoint_iterator(
      train_state, model_bundle.ckpt, workdir, num_train_steps, eval_sleep_s
  ):
    cur_train_step = int(train_state.step)
    with reporter.timed('eval'):
      valid_metrics = valid_metrics_collection.empty()
      for valid_step, batch in enumerate(valid_dataset.as_numpy_iterator()):
        batch = jax.tree.map(np.asarray, batch)
        model_outputs, new_valid_metrics = get_metrics(
            batch, flax_utils.replicate(train_state)
        )
        valid_metrics = valid_metrics.merge(
            flax_utils.unreplicate(new_valid_metrics)
        )

        _audio_and_spectrogram_summaries(
            writer,
            cur_train_step,
            'eval',
            batch,
            model_outputs.separated_audio,
            sample_rate=sample_rate_hz,
        )

        if (
            eval_steps_per_checkpoint > 0
            and valid_step >= eval_steps_per_checkpoint
        ):
          break

      # Log validation loss
      train_utils.write_metrics(
          writer, cur_train_step, valid_metrics.compute(prefix='valid')
      )
    writer.flush()


def export_tf_model(
    model_bundle: train_utils.ModelBundle,
    train_state: train_utils.TrainState,
    workdir: str,
    num_train_steps: int,
    frame_size: int,
    eval_sleep_s: int = EVAL_LOOP_SLEEP_S,
):
  """Write a TFLite flatbuffer.

  Args:
    model_bundle: The model bundle.
    train_state: The train state.
    workdir: Where to place the exported model.
    num_train_steps: Number of training steps.
    frame_size: Frame size for input audio. The exported model will take inputs
      with shape [B, T//frame_size, frame_size]. This ensures that the time
      dimension is divisible by the product of all model strides, which allows
      us to set a polymorphic time dimension. Thus, the frame_size must be
      divisible by the product of all strides in the model.
    eval_sleep_s: Number of seconds to sleep when waiting for next checkpoint.
  """
  for train_state in train_utils.checkpoint_iterator(
      train_state, model_bundle.ckpt, workdir, num_train_steps, eval_sleep_s
  ):
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

    logging.info('Creating converted_model...')
    converted_model = export_utils.Jax2TfModelWrapper(
        infer_fn, variables, [None, None, frame_size], False
    )
    logging.info('Exporting converted_model...')
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
  valid_dataset = None
  train_dataset = None
  if mode == 'train':
    train_dataset, dataset_info = data_utils.get_dataset(
        is_train=True,
        tf_data_service_address=tf_data_service_address,
        **config.train_dataset_config,
    )
  elif mode == 'eval':
    valid_dataset, dataset_info = data_utils.get_dataset(
        **config.eval_dataset_config
    )
  elif mode == 'export':
    valid_dataset = None
    dataset_info = None
  else:
    raise ValueError(f'Unknown run mode: "{mode}"!')

  if (
      dataset_info is not None
      and dataset_info.features['audio'].sample_rate != config.sample_rate_hz
  ):
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
        loss_fn=config.loss_fn,
        logdir=workdir,
        **config.train_config,
    )
  elif mode == 'eval':
    evaluate(
        model_bundle,
        train_state,
        valid_dataset,
        loss_fn=config.loss_fn,
        workdir=workdir,
        **config.eval_config,
    )
  elif mode == 'export':
    export_tf_model(model_bundle, train_state, workdir, **config.export_config)
