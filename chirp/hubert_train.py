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
from chirp.models import class_average
from chirp.models import hubert
from chirp.models import metrics
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


def filter_loss(loss, keep_inds):
  """Filters `loss` based on `keep_inds`.

  Args:
    loss: [bsz, sz]. The loss for each timestep in each batch sample.
    keep_inds: [bsz, sz]. A mask that determines which timesteps to consider.

  Returns:
    loss_filtered: [bsz, sz]. A jnp.array that is such that averaging over it
    yields the same result as averaging over loss[keep_inds], which we can't
    compute directly due to a concretization error.
  """
  # First, compute the mean of the entries to keep, as per `keep_inds`.
  loss_filtered_zeros = jnp.where(jnp.squeeze(keep_inds), loss, 0)
  mean_of_kept = jnp.sum(loss_filtered_zeros) / jnp.sum(keep_inds)
  # Now replace the entries of `loss` that we don't want to keep by this mean.
  loss_filtered = jnp.where(keep_inds, loss, mean_of_kept)
  return loss_filtered


def filtered_hubert_loss_from_outputs(outputs: hubert.ModelOutputs,
                                      keep_inds: jnp.ndarray,
                                      **unused_kwargs) -> jnp.ndarray:
  """Cross entropy from model outputs for the given subset of `keep_inds`."""
  logits = outputs.logits
  targets = outputs.targets

  # [bsz, sz, num classes].
  loss = optax.softmax_cross_entropy(logits, targets)
  # [bsz, sz].
  loss_filtered = filter_loss(loss, keep_inds)
  return loss_filtered


def hubert_loss_from_outputs(outputs: hubert.ModelOutputs, alpha: float,
                             **unused_kwargs) -> jnp.ndarray:
  """Cross entropy computed from model outputs."""
  mask_idc = outputs.mask_idc
  # Compute the loss on the unmasked and masked frames separately.
  loss_u = filtered_hubert_loss_from_outputs(outputs,
                                             jnp.where(mask_idc, False, True))
  loss_m = filtered_hubert_loss_from_outputs(outputs,
                                             jnp.where(mask_idc, True, False))
  return alpha * loss_m + (1 - alpha) * loss_u


def quantizer_loss(outputs: hubert.ModelOutputs,
                   **unused_kwargs) -> jnp.ndarray:
  """Get quantization loss from model outputs."""
  # [batch_size, num frames, embed dim].
  quant_loss = outputs.quantization_loss
  quant_loss = jnp.squeeze(jnp.mean(quant_loss, -1))
  return quant_loss


def taxonomy_cross_entropy(outputs: hubert.ModelOutputs, label: jnp.ndarray,
                           genus: jnp.ndarray, family: jnp.ndarray,
                           order: jnp.ndarray, taxonomy_loss_weight: float,
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


def supervised_loss(outputs: hubert.ModelOutputs, label: jnp.ndarray,
                    genus: jnp.ndarray, family: jnp.ndarray, order: jnp.ndarray,
                    taxonomy_loss_weight: float,
                    **unused_kwargs) -> jnp.ndarray:
  del unused_kwargs
  loss = taxonomy_cross_entropy(outputs, label, genus, family, order,
                                taxonomy_loss_weight)  # [bsz].
  # Make it [bsz, sz] so that it can be element-wise added to other losses.
  sz = outputs.logits.shape[-2]
  loss = jnp.repeat(jnp.expand_dims(loss, axis=-1), axis=-1, repeats=sz)
  return loss


def keyed_cross_entropy(key: str, outputs: hubert.ModelOutputs,
                        **kwargs) -> Optional[jnp.ndarray]:
  """Cross entropy for the specified taxonomic label set."""
  mean = jnp.mean(
      optax.sigmoid_binary_cross_entropy(getattr(outputs, key), kwargs[key]),
      axis=-1)
  return mean


def keyed_map(key: str, outputs: hubert.ModelOutputs,
              **kwargs) -> Optional[jnp.ndarray]:
  return metrics.average_precision(
      scores=getattr(outputs, key), labels=kwargs[key])


def keyed_cmap(key: str, outputs: hubert.ModelOutputs,
               **kwargs) -> Optional[jnp.ndarray]:
  return metrics.average_precision(
      scores=getattr(outputs, key), labels=kwargs[key]), kwargs[key]


def final_loss(outputs: hubert.ModelOutputs, alpha: float,
               quant_loss_mult: float,
               **kwargs_for_supervised) -> Optional[jnp.ndarray]:
  """Get the final loss to use for training."""
  quant_loss = quantizer_loss(outputs)
  hubert_loss = hubert_loss_from_outputs(outputs, alpha)
  # The gradients from this supervised loss don't flow into the representations.
  readout_loss = supervised_loss(outputs, **kwargs_for_supervised)
  return quant_loss_mult * quant_loss + hubert_loss + readout_loss


def cluster_targets_metrics(outputs: hubert.ModelOutputs, key: str,
                            **unused_kwargs) -> Optional[jnp.ndarray]:
  """Get the final loss to use for training."""
  del unused_kwargs
  assert key in [
      "n_masked_per_sample", "n_per_cluster", "max_per_cluster",
      "min_per_cluster", "h_diversity"
  ]
  # targets is [bsz, sz, nc].
  targets = outputs.targets
  mask_idc = outputs.mask_idc
  n_masked_per_sample = jnp.sum(mask_idc, axis=1)  # [bsz].
  nc = targets.shape[-1]
  targets = jnp.reshape(targets, (-1, nc))  # [bsz * sz, nc].
  n_per_cluster = jnp.sum(targets, axis=0)  # [nc].
  max_per_cluster = jnp.max(n_per_cluster)
  min_per_cluster = jnp.min(n_per_cluster)
  diversity = jnp.mean(targets, axis=0)  # [nc]
  h_diversity = -jnp.sum(diversity * jnp.log2(diversity + 1e-8))
  ret = {
      "n_masked_per_sample": n_masked_per_sample,
      "n_per_cluster": n_per_cluster,
      "max_per_cluster": max_per_cluster,
      "min_per_cluster": min_per_cluster,
      "h_diversity": h_diversity
  }
  return ret[key]


def make_metrics_collection(prefix: str, alpha: float, quant_loss_mult: float):
  """Create metrics collection."""
  metrics_dict = {
      "hubert_loss":
          clu_metrics.Average.from_fun(
              functools.partial(hubert_loss_from_outputs, alpha=alpha)),
      "quantizer_loss":
          clu_metrics.Average.from_fun(quantizer_loss),
      "supervised_loss":
          clu_metrics.Average.from_fun(supervised_loss),
      "loss":
          clu_metrics.Average.from_fun(
              functools.partial(
                  final_loss, alpha=alpha, quant_loss_mult=quant_loss_mult)),
  }

  # Debugging info:
  metrics_dict.update({
      "n_masked_per_sample":
          clu_metrics.Average.from_fun(
              functools.partial(
                  cluster_targets_metrics, key="n_masked_per_sample")),
      "n_per_cluster":
          clu_metrics.Average.from_fun(
              functools.partial(cluster_targets_metrics, key="n_per_cluster")),
      "max_per_cluster":
          clu_metrics.Average.from_fun(
              functools.partial(cluster_targets_metrics,
                                key="max_per_cluster")),
      "min_per_cluster":
          clu_metrics.Average.from_fun(
              functools.partial(cluster_targets_metrics,
                                key="min_per_cluster")),
      "h_diversity":
          clu_metrics.Average.from_fun(
              functools.partial(cluster_targets_metrics, key="h_diversity")),
  })

  taxo_keys = ["label", "genus", "family", "order"]
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


def initialize_model(dataset_info: tfds.core.DatasetInfo,
                     model_config: config_dict.ConfigDict, rng_seed: int,
                     input_size: int, learning_rate: float, workdir: str,
                     num_train_steps: int):
  """Creates model for training, eval, or inference."""
  # Initialize random number generator
  key = random.PRNGKey(rng_seed)

  # Load model
  model_init_key, mask_key = random.split(key)
  num_classes = {
      k: dataset_info.features[k].num_classes
      for k in ("label", "genus", "family", "order")
  }
  model = hubert.HuBERTModel(num_classes=num_classes, **model_config)
  variables = model.init(
      model_init_key,
      jnp.zeros((1, input_size)),
      train=False,
      mask_key=mask_key)
  model_state, params = variables.pop("params")

  # Initialize optimizer, and the learning rate schedule.
  linear_increase = optax.linear_schedule(
      init_value=0.,
      end_value=learning_rate,
      transition_steps=int(num_train_steps / 2),
      transition_begin=0)

  optimizer = optax.adam(learning_rate=linear_increase)
  opt_state = optimizer.init(params)

  # Load checkpoint
  ckpt = checkpoint.MultihostCheckpoint(workdir)
  train_state = TrainState(
      step=0, params=params, opt_state=opt_state, model_state=model_state)
  train_state = ckpt.restore_or_initialize(train_state)
  return ModelBundle(model, optimizer, key, ckpt), train_state


def train(model_bundle, train_state, train_dataset, num_train_steps: int,
          logdir: str, log_every_steps: int, checkpoint_every_steps: int,
          num_quantizer_pretrain_steps: int) -> None:
  """Train a model.

  Args:
    model_bundle: Static objects for conducting the experiment.
    train_state: Initial TrainState.
    train_dataset: Training dataset.
    num_train_steps: The number of training steps.
    logdir: Directory to use for logging.
    log_every_steps: Write the training minibatch loss.
    checkpoint_every_steps: Checkpoint the model and training state.
    num_quantizer_pretrain_steps: The number of steps to train the quantizer
      only before begining to train all parameters end-to-end.
  """
  train_iterator = train_dataset.as_numpy_iterator()
  train_metrics_collection = make_metrics_collection(
      "train___", model_bundle.model.alpha, model_bundle.model.quant_loss_mult)

  # Define update step for HuBERT (and supervised readout).
  @functools.partial(jax.pmap, axis_name="batch")
  def update_step(key, batch, train_state, mask_key):

    dropout_key, low_pass_key = random.split(key)

    def step(params, model_state):
      variables = {"params": params, **model_state}
      x = jnp.squeeze(batch["audio"])
      model_outputs, model_state = model_bundle.model.apply(
          variables,
          x,
          train=True,
          mask_key=mask_key,
          mutable=list(model_state.keys()),
          rngs={
              "dropout": dropout_key,
              "low_pass": low_pass_key,
          })
      train_metrics = train_metrics_collection.gather_from_model_output(
          outputs=model_outputs,
          label=batch["label"],
          genus=batch["genus"],
          family=batch["family"],
          order=batch["order"],
          taxonomy_loss_weight=model_bundle.model.taxonomy_loss_weight).compute(
          )
      loss = train_metrics["train___loss"]
      return loss, (train_metrics, model_state)

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

  # Define update step for quantizer.
  @functools.partial(jax.pmap, axis_name="batch")
  def update_quantizer_step(key, batch, train_state, mask_key):

    dropout_key, low_pass_key = random.split(key)

    def step(params, model_state):
      variables = {"params": params, **model_state}
      x = jnp.squeeze(batch["audio"])
      model_outputs, model_state = model_bundle.model.apply(
          variables,
          x,
          train=True,
          mask_key=mask_key,
          mutable=list(model_state.keys()),
          rngs={
              "dropout": dropout_key,
              "low_pass": low_pass_key,
          })
      train_metrics = train_metrics_collection.gather_from_model_output(
          outputs=model_outputs,
          label=batch["label"],
          genus=batch["genus"],
          family=batch["family"],
          order=batch["order"],
          taxonomy_loss_weight=model_bundle.model.taxonomy_loss_weight).compute(
          )
      loss = train_metrics["train___quantizer_loss"]
      return loss, (train_metrics, model_state)

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
      mask_key, key = random.split(key)

      mask_key = random.split(mask_key, num=jax.device_count())
      step_key = random.split(step_key, num=jax.device_count())

      if step < num_quantizer_pretrain_steps:
        # Train only the quantizer.
        train_metrics, train_state = update_quantizer_step(
            step_key, batch, train_state, mask_key)
        # Delete this debugging stuff:
        # print("train_state.params codes {}".format(
        #     train_state.params["quantizer"]["codebook"]))
        # print("train_state.params conv net {}".format(
        #     train_state.params["early_feature_extractor"]))
        # print("train_state.params keys {}".format(train_state.params.keys()))
      else:
        train_metrics, train_state = update_step(step_key, batch, train_state,
                                                 mask_key)

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


def evaluate(model_bundle: ModelBundle,
             train_state: TrainState,
             valid_dataset: tf.data.Dataset,
             writer: metric_writers.MetricWriter,
             reporter: periodic_actions.ReportProgress,
             eval_steps_per_checkpoint: Optional[int] = None):
  """Run evaluation."""
  valid_metrics = make_metrics_collection("valid___", model_bundle.model.alpha,
                                          model_bundle.model.quant_loss_mult)

  @functools.partial(jax.pmap, axis_name="batch")
  def update_metrics(valid_metrics, batch, train_state):
    variables = {"params": train_state.params, **train_state.model_state}
    model_outputs = model_bundle.model.apply(
        variables, batch["audio"], train=False, mask_key=None)
    return valid_metrics.merge(
        valid_metrics.gather_from_model_output(
            outputs=model_outputs,
            label=batch["label"],
            genus=batch["genus"],
            family=batch["family"],
            order=batch["order"],
            taxonomy_loss_weight=model_bundle.model.taxonomy_loss_weight,
            axis_name="batch"))

  step = int(flax_utils.unreplicate(train_state.step))
  with reporter.timed("eval"):
    valid_metrics = flax_utils.replicate(valid_metrics.empty())
    for s, batch in enumerate(valid_dataset.as_numpy_iterator()):
      batch = jax.tree_map(np.asarray, batch)
      valid_metrics = update_metrics(valid_metrics, batch, train_state)
      if eval_steps_per_checkpoint is not None and s >= eval_steps_per_checkpoint:
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
                  eval_steps_per_checkpoint: Optional[int] = None,
                  tflite_export: bool = False,
                  input_size: Optional[int] = None,
                  eval_sleep_s: int = EVAL_LOOP_SLEEP_S):
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
      time.sleep(eval_sleep_s)
      continue
    try:
      train_state = ckpt.restore_or_initialize(train_state)
    except tf.errors.NotFoundError:
      logging.warning("Checkpoint %s not found in workdir %s",
                      ckpt.latest_checkpoint, workdir)
      time.sleep(eval_sleep_s)
      continue

    evaluate(model_bundle, flax_utils.replicate(train_state), valid_dataset,
             writer, reporter, eval_steps_per_checkpoint)
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
