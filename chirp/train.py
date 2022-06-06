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
import os
import time
from typing import Optional
from absl import logging
from chirp import audio_utils
from chirp.models import class_average
from chirp.models import efficientnet
from chirp.models import metrics
from chirp.models import taxonomy_model
from clu import checkpoint
from clu import metric_writers
from clu import metrics as clu_metrics
from clu import periodic_actions
import flax
from flax import linen as nn
import flax.jax_utils as flax_utils
import jax
from jax import numpy as jnp
from jax import random
from jax.experimental import jax2tf
from ml_collections import config_dict
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds


EVAL_LOOP_SLEEP_S = 30


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
  mean = jnp.mean(
      optax.sigmoid_binary_cross_entropy(getattr(outputs, key), kwargs[key]),
      axis=-1)
  return mean


def keyed_map(key: str, outputs: taxonomy_model.ModelOutputs,
              **kwargs) -> Optional[jnp.ndarray]:
  return metrics.average_precision(
      scores=getattr(outputs, key), labels=kwargs[key])


def keyed_cmap(key: str, outputs: taxonomy_model.ModelOutputs,
               **kwargs) -> Optional[jnp.ndarray]:
  return metrics.average_precision(
      scores=getattr(outputs, key), labels=kwargs[key]), kwargs[key]


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
        key + "_cmap":
            class_average.ClassAverage.from_fun(
                functools.partial(keyed_cmap, key=key)),
    })
  if taxonomy_loss_weight != 0.0:
    metrics_dict["loss"] = clu_metrics.Average.from_fun(taxonomy_cross_entropy)
  else:
    metrics_dict["loss"] = metrics_dict["label_xentropy"]
  metrics_dict = {prefix + k: v for k, v in metrics_dict.items()}
  return clu_metrics.Collection.create(**metrics_dict)


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


def parse_config(config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Parse the model configuration.

  This converts string-based configuration into the necessary objects.

  Args:
    config: The model configuration. Will be modified in-place.

  Returns:
    The modified model configuration which can be passed to the model
    constructor.
  """
  # Handle model config
  model_config = config.model_config
  with model_config.unlocked():
    if model_config.encoder_.startswith("efficientnet-"):
      model = efficientnet.EfficientNetModel(model_config.encoder_[-2:])
      model_config.encoder = efficientnet.EfficientNet(model)
    else:
      raise ValueError("unknown encoder")
    del model_config.encoder_

  # Handle melspec config
  melspec_config = model_config.melspec_config
  with melspec_config.unlocked():
    # TODO(bartvm): Add scaling config for hyperparameter search
    melspec_config.sample_rate_hz = config.sample_rate_hz
    if melspec_config.scaling == "pcen":
      melspec_config.scaling_config = audio_utils.PCENScalingConfig()
    elif melspec_config.scaling == "log":
      melspec_config.scaling_config = audio_utils.LogScalingConfig()
    elif melspec_config.scaling == "raw":
      melspec_config.scaling_config = None
    del melspec_config.scaling

  with config.unlocked():
    config.input_size = (
        config.data_config.window_size_s * config.sample_rate_hz)
  if config.eval_config.tflite_export and not melspec_config.use_tf_stft:
    logging.warning(
        "TFLite export will probably fail if using the JAX stft op.")
  return config


def initialize_model(dataset_info: tfds.core.DatasetInfo,
                     data_config: config_dict.ConfigDict,
                     model_config: config_dict.ConfigDict, rng_seed: int,
                     learning_rate: float, workdir: str):
  """Creates model for training, eval, or inference."""
  # Initialize random number generator
  key = random.PRNGKey(rng_seed)

  # Load model
  model_init_key, key = random.split(key)
  num_classes = {
      k: dataset_info.features[k].num_classes
      for k in ("label", "genus", "family", "order")
  }
  model = taxonomy_model.TaxonomyModel(num_classes=num_classes, **model_config)
  variables = model.init(
      model_init_key,
      jnp.zeros((1, dataset_info.features["audio"].sample_rate *
                 data_config.window_size_s)),
      train=False)
  model_state, params = variables.pop("params")

  # Initialize optimizer
  optimizer = optax.adam(learning_rate=learning_rate)
  opt_state = optimizer.init(params)

  # Load checkpoint
  ckpt = checkpoint.MultihostCheckpoint(workdir)
  train_state = TrainState(
      step=0, params=params, opt_state=opt_state, model_state=model_state)
  train_state = ckpt.restore_or_initialize(train_state)
  return ModelBundle(model, optimizer, key, ckpt), train_state


def train(model_bundle, train_state, train_dataset, num_train_steps: int,
          logdir: str, log_every_steps: int,
          checkpoint_every_steps: int) -> None:
  """Train a model.

  Args:
    model_bundle: Static objects for conducting the experiment.
    train_state: Initial TrainState.
    train_dataset: Training dataset.
    num_train_steps: The number of training steps.
    logdir: Directory to use for logging.
    log_every_steps: Write the training minibatch loss.
    checkpoint_every_steps: Checkpoint the model and training state.
  """
  train_iterator = train_dataset.as_numpy_iterator()
  train_metrics_collection = make_metrics_collection(
      "train___", model_bundle.model.taxonomy_loss_weight)

  # Define update step
  @functools.partial(jax.pmap, axis_name="batch")
  def update_step(key, batch, train_state):

    dropout_key, low_pass_key = random.split(key)

    def step(params, model_state):
      variables = {"params": params, **model_state}
      model_outputs, model_state = model_bundle.model.apply(
          variables,
          batch["audio"],
          train=True,
          mutable=list(model_state.keys()),
          rngs={
              "dropout": dropout_key,
              "low_pass": low_pass_key
          })
      taxonomy_loss_weight = model_bundle.model.taxonomy_loss_weight
      train_metrics = train_metrics_collection.gather_from_model_output(
          outputs=model_outputs,
          label=batch["label"],
          genus=batch["genus"],
          family=batch["family"],
          order=batch["order"],
          taxonomy_loss_weight=taxonomy_loss_weight).compute()
      return train_metrics["train___loss"], (train_metrics, model_state)

    (_, (train_metrics, model_state)), grads = jax.value_and_grad(
        step, has_aux=True)(train_state.params, train_state.model_state)
    grads = jax.lax.pmean(grads, axis_name="batch")
    updates, opt_state = model_bundle.optimizer.update(grads,
                                                       train_state.opt_state)
    params = optax.apply_updates(train_state.params, updates)
    train_state = TrainState(
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
      step_key = random.split(step_key, num=jax.device_count())
      train_metrics, train_state = update_step(step_key, batch, train_state)
      train_metrics = flax_utils.unreplicate(train_metrics)

      if step % log_every_steps == 0:
        train_metrics = {
            k.replace("___", "/"): v for k, v in train_metrics.items()
        }
        writer.write_scalars(step, train_metrics)
      reporter(step)
    # Run validation loop
    if (step + 1) % checkpoint_every_steps == 0:
      with reporter.timed("checkpoint"):
        model_bundle.ckpt.save(flax_utils.unreplicate(train_state))
  writer.close()


def evaluate(model_bundle: ModelBundle,
             train_state: TrainState,
             valid_dataset: tf.data.Dataset,
             writer: metric_writers.MetricWriter,
             reporter: periodic_actions.ReportProgress,
             max_eval_steps: int = -1):
  """Run evaluation."""
  valid_metrics = make_metrics_collection(
      "valid___", model_bundle.model.taxonomy_loss_weight)

  @functools.partial(jax.pmap, axis_name="batch")
  def update_metrics(valid_metrics, batch, train_state):
    variables = {"params": train_state.params, **train_state.model_state}
    model_outputs = model_bundle.model.apply(
        variables, batch["audio"], train=False)
    return valid_metrics.merge(
        valid_metrics.gather_from_model_output(
            model_outputs=model_outputs,
            label=batch["label"],
            genus=batch["genus"],
            family=batch["family"],
            order=batch["order"],
            taxonomy_loss_weight=model_bundle.model.taxonomy_loss_weight,
            axis_name="batch"))

  step = flax_utils.unreplicate(train_state.step)
  with reporter.timed("eval"):
    valid_metrics = flax_utils.replicate(valid_metrics.empty())
    for s, batch in enumerate(valid_dataset.as_numpy_iterator()):
      batch = jax.tree_map(np.asarray, batch)
      valid_metrics = update_metrics(valid_metrics, batch, train_state)
      if max_eval_steps > 0 and s >= max_eval_steps:
        break

    # Log validation loss
    valid_metrics = flax_utils.unreplicate(valid_metrics).compute()

  valid_metrics = {k.replace("___", "/"): v for k, v in valid_metrics.items()}
  writer.write_scalars(step, valid_metrics)
  writer.flush()


def evaluate_loop(model_bundle: ModelBundle,
                  train_state: TrainState,
                  valid_dataset: tf.data.Dataset,
                  workdir: str,
                  logdir: str,
                  num_train_steps: int,
                  eval_steps_per_loop: int,
                  tflite_export: bool = False,
                  input_size: int = -1):
  """Run evaluation in a loop."""
  writer = metric_writers.create_default_writer(logdir)
  reporter = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer)
  # Initialize last_step to zero so we always run at least one eval.
  last_step = -1
  last_ckpt = ""

  while last_step < num_train_steps:
    ckpt = checkpoint.MultihostCheckpoint(workdir)
    if ckpt.latest_checkpoint == last_ckpt:
      time.sleep(EVAL_LOOP_SLEEP_S)
      continue
    try:
      train_state = ckpt.restore_or_initialize(train_state)
    except tf.errors.NotFoundError:
      logging.warning("Checkpoint %s not found in workdir %s",
                      ckpt.latest_checkpoint, workdir)
      time.sleep(EVAL_LOOP_SLEEP_S)
      continue

    evaluate(model_bundle, flax_utils.replicate(train_state), valid_dataset,
             writer, reporter, eval_steps_per_loop)
    if tflite_export:
      export_tf_lite(model_bundle, train_state, workdir, input_size)
    last_step = int(train_state.step)
    last_ckpt = ckpt.latest_checkpoint


def export_tf_lite(model_bundle: ModelBundle, train_state: TrainState,
                   workdir: str, input_size: int):
  """Write a TFLite flatbuffer."""
  variables = {"params": train_state.params, **train_state.model_state}

  def infer_fn(audio_batch):
    model_outputs = model_bundle.model.apply(
        variables, audio_batch, train=False)
    return model_outputs.label

  tf_predict = tf.function(
      jax2tf.convert(infer_fn, enable_xla=False),
      input_signature=[
          tf.TensorSpec(shape=[1, input_size], dtype=tf.float32, name="input")
      ],
      autograph=False)

  converter = tf.lite.TFLiteConverter.from_concrete_functions(
      [tf_predict.get_concrete_function()], tf_predict)

  converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
      tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
  ]
  tflite_float_model = converter.convert()

  if not tf.io.gfile.exists(workdir):
    tf.io.gfile.makedirs(workdir)
  with tf.io.gfile.GFile(os.path.join(workdir, "model.tflite"), "wb") as f:
    f.write(tflite_float_model)
