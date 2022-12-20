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

"""Training loop."""
import functools
import time
from typing import Optional, Tuple

from absl import logging
from chirp import export_utils
from chirp.data import pipeline
from chirp.models import cmap
from chirp.models import frontend
from chirp.models import metrics
from chirp.models import taxonomy_model
from chirp.taxonomy import class_utils
from chirp.train import utils
from clu import checkpoint
from clu import metric_writers
from clu import metrics as clu_metrics
from clu import periodic_actions
from flax import traverse_util
import flax.jax_utils as flax_utils
import jax
from jax import numpy as jnp
from jax import random
from jax import tree_util
from ml_collections import config_dict
import numpy as np
import optax
import tensorflow as tf

EVAL_LOOP_SLEEP_S = 30


# Metric and logging utilities
def taxonomy_cross_entropy(outputs: taxonomy_model.ModelOutputs,
                           label: jnp.ndarray, genus: jnp.ndarray,
                           family: jnp.ndarray, order: jnp.ndarray,
                           taxonomy_loss_weight: float,
                           **unused_kwargs) -> jnp.ndarray:
  """Computes mean cross entropy across taxonomic labels."""
  mean = jnp.mean(
      optax.sigmoid_binary_cross_entropy(outputs.label, label), axis=-1)
  if taxonomy_loss_weight != 0:
    mean += taxonomy_loss_weight * jnp.mean(
        optax.sigmoid_binary_cross_entropy(outputs.genus, genus), axis=-1)
    mean += taxonomy_loss_weight * jnp.mean(
        optax.sigmoid_binary_cross_entropy(outputs.family, family), axis=-1)
    mean += taxonomy_loss_weight * jnp.mean(
        optax.sigmoid_binary_cross_entropy(outputs.order, order), axis=-1)
  return mean


def keyed_cross_entropy(key: str, outputs: taxonomy_model.ModelOutputs,
                        **kwargs) -> Optional[jnp.ndarray]:
  """Cross entropy for the specified taxonomic label set."""
  cross_entropy = optax.sigmoid_binary_cross_entropy(
      getattr(outputs, key), kwargs[key])
  label_mask = kwargs.get(key + "_mask", 1)
  cross_entropy = label_mask * cross_entropy
  mean = jnp.mean(cross_entropy, axis=-1)
  return mean


def keyed_map(key: str, outputs: taxonomy_model.ModelOutputs,
              **kwargs) -> Optional[jnp.ndarray]:
  label_mask = kwargs.get(key + "_mask", None)
  return metrics.average_precision(
      scores=getattr(outputs, key), labels=kwargs[key], label_mask=label_mask)


def make_metrics_collection(prefix: str, taxonomy_loss_weight: float):
  """Create metrics collection."""
  # pylint: disable=g-long-lambda
  metrics_dict = {}
  if taxonomy_loss_weight != 0.0:
    taxo_keys = ["label", "genus", "family", "order"]
  else:
    taxo_keys = ["label"]

  for key in taxo_keys:
    metrics_dict.update({
        key + "_xentropy":
            clu_metrics.Average.from_fun(
                functools.partial(keyed_cross_entropy, key=key)),
        key + "_map":
            clu_metrics.Average.from_fun(functools.partial(keyed_map, key=key)),
    })
  if taxonomy_loss_weight != 0.0:
    metrics_dict["loss"] = clu_metrics.Average.from_fun(taxonomy_cross_entropy)
  else:
    metrics_dict["loss"] = metrics_dict["label_xentropy"]
  metrics_dict = {prefix + k: v for k, v in metrics_dict.items()}
  return clu_metrics.Collection.create(**metrics_dict)


# Projected gradient descent utilities
# TODO(bartvm): Move to separate file.
def mask_by_name(name, pytree):
  """Create a mask which is only true for leaves with the given name."""
  flat_tree = traverse_util.flatten_dict(pytree)
  mask = {k: k[-1] == name for k in flat_tree}
  return traverse_util.unflatten_dict(mask)


def project(min_value: float, max_value: float) -> optax.GradientTransformation:
  """Optax gradient transformation that projects values within a range."""

  def clip_value(updates, params):
    return tree_util.tree_map(
        lambda p, u: jnp.clip(p + u, min_value, max_value) - p, params, updates)

  return optax.stateless(clip_value)


def initialize_model(
    model_config: config_dict.ConfigDict, rng_seed: int,
    input_shape: Tuple[int, ...], learning_rate: float, workdir: str,
    target_class_list: str) -> Tuple[utils.ModelBundle, utils.TrainState]:
  """Creates model for training, eval, or inference."""
  # Initialize random number generator
  key = random.PRNGKey(rng_seed)

  # Handle lazy computation
  input_shape = tuple(s.get() if hasattr(s, "get") else s for s in input_shape)

  # Load model
  model_init_key, key = random.split(key)
  class_lists = class_utils.get_class_lists(target_class_list, True)
  model = taxonomy_model.TaxonomyModel(
      num_classes={k: v.size for (k, v) in class_lists.items()}, **model_config)
  variables = model.init(
      model_init_key, jnp.zeros((1,) + input_shape), train=False)
  model_state, params = variables.pop("params")
  # NOTE: https://github.com/deepmind/optax/issues/160
  params = params.unfreeze()

  # Initialize optimizer and handle constraints
  std_to_fwhm = jnp.sqrt(2 * jnp.log(2)) / jnp.pi
  if isinstance(model.frontend, frontend.MorletWaveletTransform):
    optimizer = optax.chain(
        optax.adam(learning_rate=learning_rate),
        optax.masked(
            project(0.0, 1.0), mask_by_name("spcen_smoothing_coef", params)),
        optax.masked(project(0.0, jnp.pi), mask_by_name("gabor_mean", params)),
        optax.masked(
            project(4 * std_to_fwhm, model.frontend.kernel_size * std_to_fwhm),
            mask_by_name("gabor_std", params)))
  else:
    optimizer = optax.adam(learning_rate=learning_rate)
  opt_state = optimizer.init(params)

  # Load checkpoint
  ckpt = checkpoint.MultihostCheckpoint(workdir)
  train_state = utils.TrainState(
      step=0, params=params, opt_state=opt_state, model_state=model_state)
  return utils.ModelBundle(model, optimizer, key, ckpt,
                           class_lists), train_state


def train(model_bundle, train_state, train_dataset, num_train_steps: int,
          logdir: str, log_every_steps: int,
          checkpoint_every_steps: int) -> None:
  """Train a model.

  Args:
    model_bundle: Static objects for conducting the experiment.
    train_state: Initial utils.TrainState.
    train_dataset: Training dataset.
    num_train_steps: The number of training steps.
    logdir: Directory to use for logging.
    log_every_steps: Write the training minibatch loss.
    checkpoint_every_steps: Checkpoint the model and training state.
  """
  train_iterator = train_dataset.as_numpy_iterator()
  train_metrics_collection = make_metrics_collection(
      "train___", model_bundle.model.taxonomy_loss_weight)

  # Forward pass and metrics
  def forward(params, key, batch, model_state):
    dropout_key, low_pass_key, patch_mask_key = random.split(key, num=3)
    variables = {"params": params, **model_state}
    kwargs = {"mask": batch["audio_mask"]} if "audio_mask" in batch else {}
    model_outputs, model_state = model_bundle.model.apply(
        variables,
        batch["audio"],
        train=True,
        mutable=list(model_state.keys()),
        rngs={
            "dropout": dropout_key,
            "low_pass": low_pass_key,
            "patch_mask": patch_mask_key,
        },
        **kwargs)
    taxonomy_loss_weight = model_bundle.model.taxonomy_loss_weight
    train_metrics = train_metrics_collection.gather_from_model_output(
        outputs=model_outputs,
        taxonomy_loss_weight=taxonomy_loss_weight,
        **batch).compute()
    return train_metrics["train___loss"], (train_metrics, model_state)

  # Define update step
  @functools.partial(jax.pmap, axis_name="batch")
  def update_step(key, batch, train_state):
    grads, (train_metrics, model_state) = jax.grad(
        forward, has_aux=True)(train_state.params, key, batch,
                               train_state.model_state)
    grads = jax.lax.pmean(grads, axis_name="batch")
    updates, opt_state = model_bundle.optimizer.update(grads,
                                                       train_state.opt_state,
                                                       train_state.params)
    params = optax.apply_updates(train_state.params, updates)
    train_state = utils.TrainState(
        step=train_state.step + 1,
        params=params,
        opt_state=opt_state,
        model_state=model_state)
    return train_metrics, train_state

  initial_step = int(train_state.step)
  train_state = flax_utils.replicate(train_state)

  # Logging
  writer = metric_writers.create_default_writer(logdir)
  reporter = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer)

  # Training and evaluation loop
  key = model_bundle.key

  for step in range(initial_step, num_train_steps + 1):
    with jax.profiler.StepTraceAnnotation("train", step_num=step):
      batch = next(train_iterator)
      step_key, key = random.split(key)
      step_key = random.split(step_key, num=jax.local_device_count())
      train_metrics, train_state = update_step(step_key, batch, train_state)
      train_metrics = flax_utils.unreplicate(train_metrics)

      if step % log_every_steps == 0:
        train_metrics = {
            k.replace("___", "/"): v for k, v in train_metrics.items()
        }
        writer.write_scalars(step, train_metrics)
      reporter(step)

    if (step + 1) % checkpoint_every_steps == 0 or step == num_train_steps:
      with reporter.timed("checkpoint"):
        model_bundle.ckpt.save(flax_utils.unreplicate(train_state))
  writer.close()


def evaluate(model_bundle: utils.ModelBundle,
             train_state: utils.TrainState,
             valid_dataset: tf.data.Dataset,
             writer: metric_writers.MetricWriter,
             reporter: periodic_actions.ReportProgress,
             eval_steps_per_checkpoint: Optional[int] = None):
  """Run evaluation."""
  valid_metrics = make_metrics_collection(
      "valid___", model_bundle.model.taxonomy_loss_weight)

  @functools.partial(jax.pmap, axis_name="batch")
  def update_metrics(valid_metrics, batch, train_state):
    variables = {"params": train_state.params, **train_state.model_state}
    kwargs = {"mask": batch["audio_mask"]} if "audio_mask" in batch else {}
    model_outputs = model_bundle.model.apply(
        variables, batch["audio"], train=False, **kwargs)
    return model_outputs, valid_metrics.merge(
        valid_metrics.gather_from_model_output(
            outputs=model_outputs,
            taxonomy_loss_weight=model_bundle.model.taxonomy_loss_weight,
            axis_name="batch",
            **batch))

  step = int(flax_utils.unreplicate(train_state.step))
  if model_bundle.model.taxonomy_loss_weight > 0:
    cmap_metrics = cmap.make_cmap_metrics_dict(
        ("label", "genus", "family", "order"))
  else:
    cmap_metrics = cmap.make_cmap_metrics_dict(("label",))
  with reporter.timed("eval"):
    valid_metrics = flax_utils.replicate(valid_metrics.empty())
    for s, batch in enumerate(valid_dataset.as_numpy_iterator()):
      batch = jax.tree_map(np.asarray, batch)
      model_outputs, valid_metrics = update_metrics(valid_metrics, batch,
                                                    train_state)
      cmap_metrics = cmap.update_cmap_metrics_dict(cmap_metrics, model_outputs,
                                                   batch)
      if eval_steps_per_checkpoint is not None and s >= eval_steps_per_checkpoint:
        break

    # Log validation loss
    valid_metrics = flax_utils.unreplicate(valid_metrics).compute()

  valid_metrics = {k.replace("___", "/"): v for k, v in valid_metrics.items()}
  cmap_metrics = flax_utils.unreplicate(cmap_metrics)
  for key in cmap_metrics:
    valid_metrics[f"valid/{key}_cmap"] = cmap_metrics[key].compute()
  writer.write_scalars(step, valid_metrics)
  writer.flush()


def evaluate_loop(model_bundle: utils.ModelBundle,
                  train_state: utils.TrainState,
                  valid_dataset: tf.data.Dataset,
                  workdir: str,
                  logdir: str,
                  num_train_steps: int,
                  eval_steps_per_checkpoint: Optional[int] = None,
                  tflite_export: bool = False,
                  input_shape: Optional[Tuple[int, ...]] = None,
                  eval_sleep_s: int = EVAL_LOOP_SLEEP_S):
  """Run evaluation in a loop."""
  writer = metric_writers.create_default_writer(logdir)
  reporter = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer)
  # Initialize last_step to -1 so we always run at least one eval.
  last_step = -1
  last_ckpt = ""

  # Handle lazy computation
  input_shape = tuple(s.get() if hasattr(s, "get") else s for s in input_shape)

  while last_step < num_train_steps:
    ckpt = checkpoint.MultihostCheckpoint(workdir)
    next_ckpt = ckpt.get_latest_checkpoint_to_restore_from()
    if next_ckpt is None or next_ckpt == last_ckpt:
      time.sleep(eval_sleep_s)
      continue
    try:
      train_state = ckpt.restore(train_state, next_ckpt)
    except tf.errors.NotFoundError:
      logging.warning("Checkpoint %s not found in workdir %s",
                      ckpt.latest_checkpoint, workdir)
      time.sleep(eval_sleep_s)
      continue

    evaluate(model_bundle, flax_utils.replicate(train_state), valid_dataset,
             writer, reporter, eval_steps_per_checkpoint)
    if tflite_export:
      export_tf(model_bundle, train_state, workdir, input_shape)
    last_step = int(train_state.step)
    last_ckpt = next_ckpt


def export_tf(model_bundle: utils.ModelBundle, train_state: utils.TrainState,
              workdir: str, input_shape: Tuple[int, ...]):
  """Export SavedModel and TFLite."""
  variables = {"params": train_state.params, **train_state.model_state}

  def infer_fn(audio_batch, variables):
    model_outputs = model_bundle.model.apply(
        variables, audio_batch, train=False)
    return model_outputs.label, model_outputs.embedding

  # Note: Polymorphic batch size currently isn't working with the STFT op,
  # so we provide a static batch size.
  converted_model = export_utils.Jax2TfModelWrapper(infer_fn, variables,
                                                    (1,) + input_shape, False)
  converted_model.export_converted_model(workdir, train_state.step,
                                         model_bundle.class_lists)


def run(mode: str, config: config_dict.ConfigDict, workdir: str,
        tf_data_service_address: str) -> None:
  """Run the experiment."""
  if mode == "train":
    train_dataset, dataset_info = pipeline.get_dataset(
        is_train=True,
        tf_data_service_address=tf_data_service_address,
        **config.train_dataset_config)
  elif mode == "eval":
    valid_dataset, dataset_info = pipeline.get_dataset(
        **config.eval_dataset_config)
  if dataset_info.features["audio"].sample_rate != config.sample_rate_hz:
    raise ValueError(
        "Dataset sample rate must match config sample rate. To address this, "
        "need to set the sample rate in the config to {}.".format(
            dataset_info.features["audio"].sample_rate))

  model_bundle, train_state = initialize_model(
      workdir=workdir, **config.init_config)
  if mode == "train":
    train_state = model_bundle.ckpt.restore_or_initialize(train_state)
    train(
        model_bundle,
        train_state,
        train_dataset,
        logdir=workdir,
        **config.train_config)
  elif mode == "eval":
    evaluate_loop(
        model_bundle,
        train_state,
        valid_dataset,
        workdir=workdir,
        logdir=workdir,
        **config.eval_config)
