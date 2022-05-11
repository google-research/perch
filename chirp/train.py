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
from chirp import audio_utils
from chirp.data import pipeline
from chirp.models import class_average
from chirp.models import efficientnet
from chirp.models import metrics
from chirp.models import taxonomy_model
from clu import checkpoint
from clu import metric_writers
from clu import metrics as clu_metrics
from clu import periodic_actions
import flax
import jax
from jax import numpy as jnp
from jax import random
from ml_collections import config_dict
import optax


@flax.struct.dataclass
class TrainState:
  step: int
  params: flax.core.scope.VariableDict
  opt_state: optax.OptState
  model_state: flax.core.scope.FrozenVariableDict


# Metrics


def mean_cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
  return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels), axis=-1)


def map_(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
  return metrics.average_precision(scores=logits, labels=labels)


def cmap(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
  return metrics.average_precision(scores=logits, labels=labels), labels


@flax.struct.dataclass
class ValidationMetrics(clu_metrics.Collection):
  valid_loss: clu_metrics.Average.from_fun(mean_cross_entropy)
  valid_map: clu_metrics.Average.from_fun(map_)
  valid_cmap: class_average.ClassAverage.from_fun(cmap)


@flax.struct.dataclass
class TrainingMetrics(clu_metrics.Collection):
  train_loss: clu_metrics.LastValue.from_fun(mean_cross_entropy)
  train_map: clu_metrics.LastValue.from_fun(map_)


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
    if melspec_config.scaling == "pcen":
      melspec_config.scaling_config = audio_utils.PCENScalingConfig()
    elif melspec_config.scaling == "log":
      melspec_config.scaling_config = audio_utils.LogScalingConfig()
    elif melspec_config.scaling == "raw":
      melspec_config.scaling_config = None
    del melspec_config.scaling
  return config


def train_and_evaluate(batch_size: int, num_train_steps: int, rng_seed: int,
                       learning_rate: float, logdir: str, workdir: str,
                       log_every_steps: int, checkpoint_every_steps: int,
                       eval_every_steps: int,
                       model_config: config_dict.ConfigDict,
                       data_config: config_dict.ConfigDict) -> None:
  """Train a model.

  Args:
    batch_size: The batch size.
    num_train_steps: The number of training steps.
    rng_seed: The seed to use for parameter initialization and dropout.
    learning_rate: The learning rate to use for the Adam optimizer.
    logdir: Directory to use for logging.
    workdir: Directory to use for checkpointing.
    log_every_steps: Write the training minibatch loss.
    checkpoint_every_steps: Checkpoint the model and training state.
    eval_every_steps: Evaluate on the validation set.
    model_config: The model configuration.
    data_config: Data loading configuration.
  """
  # Load dataset
  train_dataset, dataset_info = pipeline.get_dataset(
      "train", batch_size=batch_size, **data_config)
  valid_dataset, _ = pipeline.get_dataset(
      "test_caples", batch_size=batch_size, **data_config)
  train_iterator = train_dataset.as_numpy_iterator()

  with model_config.unlocked():
    model_config.melspec_config.sample_rate_hz = dataset_info.features[
        "audio"].sample_rate

  # Initialize random number generator
  key = random.PRNGKey(rng_seed)

  # Load model
  model_init_key, key = random.split(key)
  model = taxonomy_model.TaxonomyModel(
      num_classes=dataset_info.features["label"].num_classes, **model_config)
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
  ckpt = checkpoint.Checkpoint(workdir)
  train_state = TrainState(
      step=0, params=params, opt_state=opt_state, model_state=model_state)
  train_state = ckpt.restore_or_initialize(train_state)

  # Define update step
  @jax.jit
  def update_step(key, batch, train_state):

    dropout_key, low_pass_key = random.split(key)

    def step(params, model_state):
      variables = {"params": params, **model_state}
      logits, model_state = model.apply(
          variables,
          batch["audio"],
          train=True,
          mutable=list(model_state.keys()),
          rngs={
              "dropout": dropout_key,
              "low_pass": low_pass_key
          })
      train_metrics = TrainingMetrics.single_from_model_output(
          logits=logits, labels=batch["label"]).compute()
      return train_metrics["train_loss"], (train_metrics, model_state)

    (_, (train_metrics, model_state)), grads = jax.value_and_grad(
        step, has_aux=True)(train_state.params, train_state.model_state)
    updates, opt_state = optimizer.update(grads, train_state.opt_state)
    params = optax.apply_updates(train_state.params, updates)
    train_state = TrainState(
        step=train_state.step + 1,
        params=params,
        opt_state=opt_state,
        model_state=model_state)
    return train_metrics, train_state

  @jax.jit
  def update_metrics(valid_metrics, batch, train_state):
    variables = {"params": train_state.params, **train_state.model_state}
    logits = model.apply(variables, batch["audio"], train=False)
    return valid_metrics.merge(
        ValidationMetrics.single_from_model_output(
            logits=logits, labels=batch["label"]))

  # Logging
  writer = metric_writers.create_default_writer(logdir)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer)

  # Training and evaluation loop
  while train_state.step < num_train_steps:
    with jax.profiler.StepTraceAnnotation("train", step_num=train_state.step):
      batch = next(train_iterator)
      step_key, key = random.split(key)
      train_metrics, train_state = update_step(step_key, batch, train_state)

      if train_state.step % log_every_steps == 0:
        writer.write_scalars(int(train_state.step), train_metrics)
      report_progress(int(train_state.step))
    # Run validation loop
    if train_state.step % checkpoint_every_steps == 0:
      with report_progress.timed("checkpoint"):
        ckpt.save(train_state)
    if train_state.step % eval_every_steps == 0:
      # TODO(bartvm): Split eval into separate job for larger validation sets
      with report_progress.timed("eval"):
        valid_metrics = ValidationMetrics.empty()
        for batch in valid_dataset.as_numpy_iterator():
          valid_metrics = update_metrics(valid_metrics, batch, train_state)

        # Log validation loss
        writer.write_scalars(int(train_state.step), valid_metrics.compute())
  writer.close()
