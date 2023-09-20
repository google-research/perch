# coding=utf-8
# Copyright 2023 The Perch Authors.
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
from typing import Callable, Sequence

from absl import logging
from chirp import export_utils
from chirp.data import utils as data_utils
from chirp.models import metrics
from chirp.models import output
from chirp.models import taxonomy_model
from chirp.taxonomy import class_utils
from chirp.train import utils
from clu import checkpoint
from clu import metric_writers
from clu import metrics as clu_metrics
from clu import periodic_actions
import flax
import flax.jax_utils as flax_utils
import jax
from jax import numpy as jnp
from jax import random
from ml_collections import config_dict
import numpy as np
import optax
import tensorflow as tf

EVAL_LOOP_SLEEP_S = 30


def get_keyed_map_fn(key):
  def _map(**kwargs):
    return metrics.average_precision(
        scores=kwargs[f"{key}_logits"],
        labels=kwargs[key],
        label_mask=kwargs.get(f"{key}_mask", None),
    )

  return _map


def get_train_metrics(
    keys: list[str], num_labels: dict[str, int]
) -> dict[str, type[clu_metrics.Metric]]:
  """Create a collection of metrics with cross-entropy and average precision."""

  metrics_ = {"loss": clu_metrics.Average.from_output("loss")}
  for key in keys:
    metrics_[f"{key}_loss"] = utils.MultiAverage.create(
        num_labels[key]
    ).from_output(f"{key}_loss")
    metrics_[f"{key}_map"] = clu_metrics.Average.from_fun(get_keyed_map_fn(key))

  return metrics_


def initialize_model(
    model_config: config_dict.ConfigDict,
    rng_seed: int,
    input_shape: Sequence[int],
    learning_rate: float,
    workdir: str,
    target_class_list: str,
    optimizer: optax.GradientTransformation | None = None,
    for_inference: bool = False,
) -> tuple[utils.ModelBundle, utils.TrainState]:
  """Creates model for training, eval, or inference.

  Args:
    model_config: A config dict of the model parameters.
    rng_seed: Used to see the random number generator.
    input_shape: Shape of the model inputs.
    learning_rate: The learning rate to use for training.
    workdir: The directory the checkpoint is stored in.
    target_class_list: A list of target classes for the classifier output.
    optimizer: The optimizer to use during training. Optional for when loading
      pre-trained models for inference.
    for_inference: Indicates whether the model is being initialized for
      inference (if false, initialzed for training).

  Note: learning_rate is unused (it's expected to be used in constructing the
    `optimizer` argument), but it's left part of the function signature for
    backwards compatibility with the config utils.

  Returns:
    A tuple of initialized ModelBundle and TrainState objects.
  """
  del learning_rate

  # Initialize random number generator
  key = random.PRNGKey(rng_seed)

  # Load model
  model_init_key, key = random.split(key)
  class_lists = class_utils.get_class_lists(target_class_list, True)
  model = taxonomy_model.TaxonomyModel(
      num_classes={k: len(v.classes) for (k, v) in class_lists.items()},
      **model_config,
  )
  # Ensure input_shape is a tuple for concatenation.
  input_shape = tuple(input_shape)

  variables = model.init(
      model_init_key, jnp.zeros((1,) + input_shape), train=False
  )
  model_state, params = flax.core.pop(variables, "params")
  # NOTE: https://github.com/deepmind/optax/issues/160
  params = flax.core.unfreeze(params)

  # Initialize optimizer and handle constraints
  if optimizer is None or for_inference:
    opt_state = None
    logging.info("No optimizer specified - loading model for inference.")
  else:
    opt_state = optimizer.init(params)

  # Load checkpoint
  ckpt = checkpoint.MultihostCheckpoint(workdir)
  train_state = utils.TrainState(
      step=0, params=params, opt_state=opt_state, model_state=model_state
  )
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
    loss_fn: Callable[
        [jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = optax.sigmoid_binary_cross_entropy,
) -> None:
  """Train a model.

  Args:
    model_bundle: Static objects for conducting the experiment.
    train_state: Initial utils.TrainState.
    train_dataset: Training dataset.
    num_train_steps: The number of training steps.
    logdir: Directory to use for logging.
    log_every_steps: Write the training minibatch loss.
    checkpoint_every_steps: Checkpoint the model and training state.
    loss_fn: Loss function used for training.
  """
  train_iterator = train_dataset.as_numpy_iterator()
  taxonomy_keys = ["label"]
  taxonomy_loss_weight = model_bundle.model.taxonomy_loss_weight
  if taxonomy_loss_weight != 0.0:
    taxonomy_keys += utils.TAXONOMY_KEYS
  train_metrics_collection = utils.NestedCollection.create(
      **get_train_metrics(taxonomy_keys, model_bundle.model.num_classes)
  )

  # Forward pass and metrics
  def forward(params, key, batch, model_state):
    dropout_key, low_pass_key, patch_mask_key = random.split(key, num=3)
    variables = {"params": params, **model_state}
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
    )
    losses = utils.taxonomy_loss(
        outputs=model_outputs,
        taxonomy_loss_weight=taxonomy_loss_weight,
        loss_fn=loss_fn,
        **batch,
    )
    train_metrics = train_metrics_collection.gather_from_model_output(
        **output.logits(model_outputs),
        **losses,
        **batch,
    )
    return jnp.mean(losses["loss"]), (train_metrics, model_state)

  # Define update step
  @functools.partial(jax.pmap, axis_name="batch")
  def update_step(key, batch, train_state):
    grads, (train_metrics, model_state) = jax.grad(forward, has_aux=True)(
        train_state.params, key, batch, train_state.model_state
    )
    grads = jax.lax.pmean(grads, axis_name="batch")
    updates, opt_state = model_bundle.optimizer.update(
        grads, train_state.opt_state, train_state.params
    )
    params = optax.apply_updates(train_state.params, updates)
    train_state = utils.TrainState(
        step=train_state.step + 1,
        params=params,
        opt_state=opt_state,
        model_state=model_state,
    )
    return train_metrics, train_state

  initial_step = int(train_state.step)
  train_state = flax_utils.replicate(train_state)

  # Logging
  writer = metric_writers.create_default_writer(logdir)
  reporter = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer
  )

  # Training and evaluation loop
  key = model_bundle.key

  for step in range(initial_step, num_train_steps + 1):
    with jax.profiler.StepTraceAnnotation("train", step_num=step):
      batch = next(train_iterator)
      step_key, key = random.split(key)
      step_key = random.split(step_key, num=jax.local_device_count())
      train_metrics, train_state = update_step(step_key, batch, train_state)

      if step % log_every_steps == 0:
        train_metrics = flax_utils.unreplicate(train_metrics).compute(
            prefix="train"
        )
        utils.write_metrics(writer, step, train_metrics)
      reporter(step)

    if (step + 1) % checkpoint_every_steps == 0 or step == num_train_steps:
      with reporter.timed("checkpoint"):
        model_bundle.ckpt.save(flax_utils.unreplicate(train_state))
  writer.close()


def evaluate(
    model_bundle: utils.ModelBundle,
    train_state: utils.TrainState,
    valid_dataset: tf.data.Dataset,
    workdir: str,
    num_train_steps: int,
    loss_fn: Callable[
        [jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = optax.sigmoid_binary_cross_entropy,
    eval_steps_per_checkpoint: int | None = None,
    eval_sleep_s: int = EVAL_LOOP_SLEEP_S,
    name: str = "valid",
):
  """Run evaluation."""
  taxonomy_keys = ["label"]
  taxonomy_loss_weight = model_bundle.model.taxonomy_loss_weight
  if taxonomy_loss_weight != 0.0:
    taxonomy_keys += utils.TAXONOMY_KEYS

  # The metrics are the same as for training, but with rank-based metrics added.
  metrics_ = get_train_metrics(taxonomy_keys, model_bundle.model.num_classes)
  valid_metrics = {}
  for key in taxonomy_keys:
    valid_metrics[f"{key}_cmap"] = ((f"{key}_logits", key), metrics.cmap)
    valid_metrics[f"{key}_roc_auc"] = ((f"{key}_logits", key), metrics.roc_auc)
  metrics_["rank_metrics"] = utils.CollectingMetrics.from_funs(**valid_metrics)
  valid_metrics_collection = utils.NestedCollection.create(**metrics_)

  @functools.partial(jax.pmap, axis_name="batch")
  def get_metrics(batch, train_state):
    variables = {"params": train_state.params, **train_state.model_state}
    kwargs = {"mask": batch["audio_mask"]} if "audio_mask" in batch else {}
    model_outputs = model_bundle.model.apply(
        variables, batch["audio"], train=False, **kwargs
    )
    losses = utils.taxonomy_loss(
        outputs=model_outputs,
        taxonomy_loss_weight=taxonomy_loss_weight,
        loss_fn=loss_fn,
        **batch,
    )
    return valid_metrics_collection.gather_from_model_output(
        **output.logits(model_outputs),
        **batch,
        **losses,
    )

  @jax.jit
  def split_batch(batch):
    batch_size = batch["audio"].shape[0]
    num_devices = jax.local_device_count()
    device_batch_size = batch_size // num_devices

    def device_batch_fn(x):
      return jnp.reshape(
          x[: device_batch_size * num_devices],
          (num_devices, device_batch_size) + x.shape[1:],
      )

    def remainder_batch_fn(x):
      return x[device_batch_size * num_devices :][None]

    return (
        jax.tree_map(device_batch_fn, batch),
        jax.tree_map(remainder_batch_fn, batch),
    )

  writer = metric_writers.create_default_writer(workdir)
  reporter = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer
  )
  for train_state in utils.checkpoint_iterator(
      train_state, model_bundle.ckpt, workdir, num_train_steps, eval_sleep_s
  ):
    step = int(train_state.step)
    replicated_train_state = flax_utils.replicate(train_state)
    with reporter.timed("eval"):
      valid_metrics = valid_metrics_collection.empty()
      for s, batch in enumerate(valid_dataset.as_numpy_iterator()):
        batch = jax.tree_map(np.asarray, batch)
        # Handle device batching if it's not been handled by the data pipeliine
        # already.
        if batch["label"].ndim == 2:
          even_batch, remainder_batch = split_batch(batch)
          # It's possible for `even_batch` to be empty if the batch size is
          # smaller than the local device count (in which case all examples in
          # the batch are found in `remainder_batch`).
          if even_batch["label"].shape[1] > 0:
            new_valid_metrics = get_metrics(even_batch, replicated_train_state)
            valid_metrics = valid_metrics.merge(
                flax_utils.unreplicate(new_valid_metrics)
            )
          # It's also possible for `remainder_batch` to be empty if the batch
          # size is an exact multiple of the local device count (in which case
          # all examples in the batch are found in `even_batch`).
          if remainder_batch["label"].shape[1] > 0:
            new_valid_metrics = get_metrics(
                remainder_batch,
                # The remainder batch has shape [1, ...] rather than
                # [jax.local_device_count(), ...].
                jax.tree_map(lambda x: x[:1], replicated_train_state),
            )
            valid_metrics = valid_metrics.merge(
                flax_utils.unreplicate(new_valid_metrics)
            )
        else:
          new_valid_metrics = get_metrics(batch, replicated_train_state)
          valid_metrics = valid_metrics.merge(
              flax_utils.unreplicate(new_valid_metrics)
          )
        if (
            eval_steps_per_checkpoint is not None
            and s >= eval_steps_per_checkpoint
        ):
          break

      # Log validation loss
      utils.write_metrics(writer, step, valid_metrics.compute(prefix=name))
    writer.flush()


def export_tf_model(
    model_bundle: utils.ModelBundle,
    train_state: utils.TrainState,
    workdir: str,
    input_shape: tuple[int, ...],
    num_train_steps: int,
    eval_sleep_s: int = EVAL_LOOP_SLEEP_S,
    polymorphic_batch: bool = True,
):
  """Export SavedModel and TFLite."""
  for train_state in utils.checkpoint_iterator(
      train_state, model_bundle.ckpt, workdir, num_train_steps, eval_sleep_s
  ):
    variables = {"params": train_state.params, **train_state.model_state}

    def infer_fn(audio_batch, variables):
      model_outputs = model_bundle.model.apply(
          variables, audio_batch, train=False
      )
      return model_outputs.label, model_outputs.embedding

    if polymorphic_batch:
      shape = (None,) + input_shape
    else:
      shape = (1,) + input_shape
    converted_model = export_utils.Jax2TfModelWrapper(
        infer_fn, variables, shape, False
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
  if mode.startswith("eval_"):
    mode, name = mode.split("_", maxsplit=1)
    config.eval_dataset_config = getattr(config.eval_dataset_config, name)
  else:
    name = "valid"

  if mode == "train":
    train_dataset, dataset_info = data_utils.get_dataset(
        is_train=True,
        tf_data_service_address=tf_data_service_address,
        **config.train_dataset_config,
    )
    valid_dataset = None
  elif mode == "eval":
    valid_dataset, dataset_info = data_utils.get_dataset(
        **config.eval_dataset_config
    )
    train_dataset = None
  elif mode == "export":
    train_dataset, valid_dataset, dataset_info = None, None, None
  else:
    raise ValueError(f"unknown mode ({mode})")

  if (
      dataset_info is not None
      and dataset_info.features["audio"].sample_rate != config.sample_rate_hz
  ):
    raise ValueError(
        "Dataset sample rate must match config sample rate. To address this, "
        "need to set the sample rate in the config to {}.".format(
            dataset_info.features["audio"].sample_rate
        )
    )

  model_bundle, train_state = initialize_model(
      workdir=workdir, **config.init_config
  )
  if mode == "train":
    train_state = model_bundle.ckpt.restore_or_initialize(train_state)
    train(
        model_bundle,
        train_state,
        train_dataset,
        loss_fn=config.loss_fn,
        logdir=workdir,
        **config.train_config,
    )
  elif mode == "eval":
    evaluate(
        model_bundle,
        train_state,
        valid_dataset,
        loss_fn=config.loss_fn,
        workdir=workdir,
        name=name,
        **config.eval_config,
    )
  elif mode == "export":
    export_tf_model(
        model_bundle,
        train_state,
        workdir=workdir,
        **config.export_config,
    )
