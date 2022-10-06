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
from typing import Optional, List
from absl import logging
from chirp.models import cmap
from chirp.models import frontend as frontend_models
from chirp.models import hubert
from chirp.models import layers
from chirp.models import metrics
from chirp.models import quantizers
from chirp.taxonomy import class_utils
from clu import checkpoint
from clu import metric_writers
from clu import metrics as clu_metrics
from clu import periodic_actions
import flax
from flax import linen as nn
from flax import traverse_util
import flax.jax_utils as flax_utils
import jax
from jax import numpy as jnp
from jax import random
from jax import tree_util
from jax.experimental import jax2tf
from ml_collections import config_dict
import numpy as np
import optax
import tensorflow as tf

EVAL_LOOP_SLEEP_S = 30


def filter_loss(loss, keep_inds):
  """Filters `loss` based on `keep_inds`.

  Args:
    loss: [ns, bsz, sz]. The loss for each frame (sz) in each batch sample (bsz)
      for each quantizer section (ns) with ns > 1 if using product quantization.
    keep_inds: [bsz, sz]. A mask that determines which frames to consider.

  Returns:
    loss_filtered: [ns, bsz, sz]. A jnp.array that is such that averaging over
    it yields the same result as averaging over loss[keep_inds], which we can't
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

  # `logits` and `targets` are lists whose length will be the number of
  # quantizers `nq`.
  losses = []
  # Each of l and t have shape [ns, bsz, sz, nc].
  for logit, target in zip(logits, targets):
    # [ns, bsz, sz].
    loss = optax.softmax_cross_entropy(logit, target)
    # [ns, bsz, sz].
    loss_filtered = filter_loss(loss, keep_inds)
    losses.append(loss_filtered)

  # [nq, ns, bsz, sz].
  losses = jnp.stack(losses, axis=0)
  return losses


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


def quantizer_loss(outputs: hubert.ModelOutputs, quant_loss_mult: float,
                   **unused_kwargs) -> jnp.ndarray:
  """Get quantization loss from model outputs."""
  del unused_kwargs
  # [bsz, sz, csz].
  quant_loss = outputs.quantization_loss
  quant_loss = jnp.squeeze(jnp.mean(quant_loss, -1))
  # [bsz, sz].
  return quant_loss * quant_loss_mult


def taxonomy_cross_entropy(outputs: hubert.ModelOutputs, label: jnp.ndarray,
                           genus: jnp.ndarray, family: jnp.ndarray,
                           order: jnp.ndarray, taxonomy_loss_weight: float,
                           **unused_kwargs) -> jnp.ndarray:
  """Computes mean cross entropy across taxonomic labels."""

  def aggregate_losses(preds, target):
    # Iterate over the label made from different readout points.
    losses = []
    for l in preds:
      losses.append(
          jnp.mean(optax.sigmoid_binary_cross_entropy(l, target), axis=-1))
    return jnp.sum(jnp.stack(losses, axis=0), axis=0)

  mean = aggregate_losses(outputs.label, label)

  if taxonomy_loss_weight != 0:
    mean += taxonomy_loss_weight * aggregate_losses(outputs.genus, genus)
    mean += taxonomy_loss_weight * aggregate_losses(outputs.family, family)
    mean += taxonomy_loss_weight * aggregate_losses(outputs.order, order)
  return mean


def supervised_loss(outputs: hubert.ModelOutputs, label: jnp.ndarray,
                    genus: jnp.ndarray, family: jnp.ndarray, order: jnp.ndarray,
                    taxonomy_loss_weight: float, readout_loss_mult: float,
                    **unused_kwargs) -> jnp.ndarray:
  """Compute classification loss for all taxonomy heads."""
  del unused_kwargs
  loss = taxonomy_cross_entropy(outputs, label, genus, family, order,
                                taxonomy_loss_weight)  # [bsz].
  # Make it [bsz, sz] so that it can be element-wise added to other losses.
  sz = outputs.logits[0].shape[-2]
  loss = jnp.repeat(jnp.expand_dims(loss, axis=-1), axis=-1, repeats=sz)
  return loss * readout_loss_mult


def keyed_cross_entropy(key: str,
                        outputs: hubert.ModelOutputs,
                        readout_index: int = 0,
                        **kwargs) -> Optional[jnp.ndarray]:
  """Cross entropy for the specified taxonomic label set."""
  outputs = getattr(outputs, key)
  outputs = outputs[readout_index]
  mean = jnp.mean(
      optax.sigmoid_binary_cross_entropy(outputs, kwargs[key]), axis=-1)
  return mean


def keyed_map(key: str,
              outputs: hubert.ModelOutputs,
              readout_index: int = 0,
              **kwargs) -> Optional[jnp.ndarray]:
  outputs = getattr(outputs, key)
  outputs = outputs[readout_index]
  return metrics.average_precision(scores=outputs, labels=kwargs[key])


def final_loss(outputs: hubert.ModelOutputs, alpha: float,
               quant_loss_mult: float, readout_loss_mult: float,
               **kwargs_for_supervised) -> Optional[jnp.ndarray]:
  """Get the final loss to use for training."""
  # [bsz, sz].
  quant_loss = quantizer_loss(outputs, quant_loss_mult)
  # [nq, ns, bsz, sz].
  hubert_loss = hubert_loss_from_outputs(outputs, alpha)
  # [bsz, sz].
  readout_loss = supervised_loss(
      outputs, readout_loss_mult=readout_loss_mult, **kwargs_for_supervised)

  # Make the shapes match so that these losses can be added elementwise.
  nq, ns, _, _ = hubert_loss.shape
  quant_loss = jnp.repeat(jnp.expand_dims(quant_loss, 0), ns, axis=0)
  quant_loss = jnp.repeat(jnp.expand_dims(quant_loss, 0), nq, axis=0)
  readout_loss = jnp.repeat(jnp.expand_dims(readout_loss, 0), ns, axis=0)
  readout_loss = jnp.repeat(jnp.expand_dims(readout_loss, 0), nq, axis=0)

  return quant_loss + hubert_loss + readout_loss


def cluster_targets_metrics(outputs: hubert.ModelOutputs, key: str,
                            **unused_kwargs) -> Optional[jnp.ndarray]:
  """Get the final loss to use for training."""
  del unused_kwargs
  assert key.startswith(("n_masked_per_sample", "n_per_cluster",
                         "max_per_cluster", "min_per_cluster", "h_diversity"))
  # A list of [ns, bsz, sz, nc].
  all_targets = outputs.targets
  mask_idc = outputs.mask_idc
  n_masked_per_sample = jnp.sum(mask_idc, axis=1)  # [bsz].
  ret = {"n_masked_per_sample": n_masked_per_sample}
  for i, targets in enumerate(all_targets):
    nc = targets.shape[-1]
    targets = jnp.reshape(targets, (-1, nc))  # [ns * bsz * sz, nc].
    n_per_cluster = jnp.sum(targets, axis=0)  # [nc].
    max_per_cluster = jnp.max(n_per_cluster)
    min_per_cluster = jnp.min(n_per_cluster)
    diversity = jnp.mean(targets, axis=0)  # [nc]
    h_diversity = -jnp.sum(diversity * jnp.log2(diversity + 1e-8))
    ret.update({
        "n_per_cluster_{}".format(i): n_per_cluster,
        "max_per_cluster_{}".format(i): max_per_cluster,
        "min_per_cluster_{}".format(i): min_per_cluster,
        "h_diversity_{}".format(i): h_diversity
    })
  print("ret has keys {}".format(ret.keys()))
  return ret[key]


def make_metrics_collection(prefix: str, alpha: float, quant_loss_mult: float,
                            readout_loss_mult: float, readout_points: List[int],
                            quantizer_points: List[int]):
  """Create metrics collection."""
  metrics_dict = {
      "hubert_loss":
          clu_metrics.Average.from_fun(
              functools.partial(hubert_loss_from_outputs, alpha=alpha)),
      "quantizer_loss":
          clu_metrics.Average.from_fun(
              functools.partial(
                  quantizer_loss, quant_loss_mult=quant_loss_mult)),
      "supervised_loss":
          clu_metrics.Average.from_fun(
              functools.partial(
                  supervised_loss, readout_loss_mult=readout_loss_mult)),
      "loss":
          clu_metrics.Average.from_fun(
              functools.partial(
                  final_loss,
                  alpha=alpha,
                  quant_loss_mult=quant_loss_mult,
                  readout_loss_mult=readout_loss_mult)),
  }

  metrics_dict.update({
      "n_masked_per_sample":
          clu_metrics.Average.from_fun(
              functools.partial(
                  cluster_targets_metrics, key="n_masked_per_sample")),
  })
  for i, block_ind in enumerate(quantizer_points):
    block_name = "late_fs_{}".format(block_ind) if block_ind >= 0 else "earlyfs"
    metrics_dict.update({
        "n_per_cluster_{}".format(block_name):
            clu_metrics.Average.from_fun(
                functools.partial(
                    cluster_targets_metrics, key="n_per_cluster_{}".format(i))),
        "max_per_cluster_{}".format(block_name):
            clu_metrics.Average.from_fun(
                functools.partial(
                    cluster_targets_metrics,
                    key="max_per_cluster_{}".format(i))),
        "min_per_cluster_{}".format(block_name):
            clu_metrics.Average.from_fun(
                functools.partial(
                    cluster_targets_metrics,
                    key="min_per_cluster_{}".format(i))),
        "h_diversity_{}".format(block_name):
            clu_metrics.Average.from_fun(
                functools.partial(
                    cluster_targets_metrics, key="h_diversity_{}".format(i))),
    })

  taxo_keys = ["label", "genus", "family", "order"]
  for i, block_ind in enumerate(readout_points):
    for key in taxo_keys:
      metrics_dict.update({
          key + "_{}_xentropy".format(block_ind):
              clu_metrics.Average.from_fun(
                  functools.partial(
                      keyed_cross_entropy, key=key, readout_index=i)),
          key + "_{}_map".format(block_ind):
              clu_metrics.Average.from_fun(
                  functools.partial(keyed_map, key=key, readout_index=i)),
      })
  metrics_dict = {prefix + k: v for k, v in metrics_dict.items()}
  return clu_metrics.Collection.create(**metrics_dict)


def make_cmap_metrics_dict(label_names, readout_points):
  """Create a dict of empty cmap_metrics."""
  metrics_dict = {}
  for block_ind in readout_points:
    metrics_dict.update({
        label + "_{}".format(block_ind): cmap.CMAP.empty()
        for label in label_names
    })
  return metrics_dict


def update_cmap_metrics_dict(label_names, cmap_metrics, model_outputs, batch,
                             readout_points):
  """Update a dict of cmap_metrics from model_outputs and a batch."""
  for label_name in label_names:
    for i, block_ind in enumerate(readout_points):
      label_name_i = label_name + "_{}".format(block_ind)
      cmap_metrics[label_name_i] = cmap_metrics[label_name_i].merge(
          cmap.CMAP(getattr(model_outputs, label_name)[i], batch[label_name]))
  return cmap_metrics


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


# Projected gradient descent utilities
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
    model_config: config_dict.ConfigDict, rng_seed: int, input_size: int,
    learning_rate: float, start_learning_rate: float, workdir: str,
    num_train_steps: int, quantizer_config: config_dict.ConfigDict,
    base_quantizer_config: config_dict.ConfigDict,
    frontend_config: config_dict.ConfigDict,
    early_fs_config: config_dict.ConfigDict, reload_quantizer_from: str,
    target_class_list: str, **unused_kwargs):
  """Creates model for training, eval, or inference."""
  del unused_kwargs
  # Initialize random number generator
  key = random.PRNGKey(rng_seed)

  # Load model
  model_init_key, mask_key = random.split(key)
  class_lists = class_utils.get_class_lists(target_class_list, True)
  num_classes = {k: v.size for (k, v) in class_lists.items()}

  # Initialize the quantizer.
  kwargs = {
      "num_centroids": base_quantizer_config.num_centroids,
      "gamma": base_quantizer_config.gamma
  }
  quantizer_list = []
  for _ in range(len(model_config.quantizer_points)):
    base_quantizers = [
        quantizers.VectorQuantizerEnt(**kwargs)
        for _ in range(quantizer_config.num_sections)
    ]
    quantizer = quantizers.ProductQuantizer(
        num_sections=quantizer_config.num_sections,
        base_quantizers=base_quantizers)
    quantizer_list.append(quantizer)

  # Initialize the frontend.
  frontend = None
  if not frontend_config.omit_frontend:
    frontend = frontend_models.MelSpectrogram(
        features=frontend_config.features,
        stride=frontend_config.stride,
        kernel_size=frontend_config.kernel_size,
        sample_rate=frontend_config.sample_rate,
        freq_range=frontend_config.freq_range,
        scaling_config=frontend_config.scaling_config)

  # Initialize the early feature extractor.
  if early_fs_config.omit_earlyfs:
    early_fs = None
  else:
    if early_fs_config.num_frames not in [125, 63, 32, 16]:
      raise ValueError(
          "Expected early_fs_config.num_frames to be 125, 63, 32 or 16.")

    if frontend is None:
      # Their original architecture led to 500 frames which causes OOM.
      # Added an additional conv layer with stride 2 which makes it 250 instead.
      # and another, making it 125. Still was getting OOM with this with batch
      # size 128, so reduced it to 64.
      conv_layer_tuples = tuple([(512, 10, 5), (512, 3, 2), (512, 3, 2),
                                 (512, 3, 2), (512, 3, 2), (512, 2, 2),
                                 (512, 2, 2), (512, 2, 2), (512, 2, 2)])
    else:
      nf = 512
      if early_fs_config.num_frames == 125:
        # With this configuration, the number of frames is reduced from 500 to
        # 125 and the framerate is reduced from 100Hz (which the frontend
        # outputs) 25Hz.
        conv_layer_tuples = tuple([(nf, 10, 2), (nf, 3, 2), (nf, 3, 1),
                                   (nf, 3, 1), (nf, 3, 1), (nf, 2, 1),
                                   (nf, 2, 1)])
      elif early_fs_config.num_frames == 63:
        conv_layer_tuples = tuple([(nf, 10, 2), (nf, 3, 2), (nf, 3, 2),
                                   (nf, 3, 1), (nf, 3, 1), (nf, 2, 1),
                                   (nf, 2, 1)])
      elif early_fs_config.num_frames == 32:
        conv_layer_tuples = tuple([(nf, 10, 2), (nf, 3, 2), (nf, 3, 2),
                                   (nf, 3, 2), (nf, 3, 1), (nf, 2, 1),
                                   (nf, 2, 1)])
      elif early_fs_config.num_frames == 16:
        conv_layer_tuples = tuple([(nf, 10, 2), (nf, 3, 2), (nf, 3, 2),
                                   (nf, 3, 2), (nf, 3, 2), (nf, 2, 1),
                                   (nf, 2, 1)])
    early_fs = layers.EarlyFeatureExtractor(
        dropout_prob=early_fs_config.dropout_prob,
        activation=early_fs_config.activation,
        conv_layer_tuples=conv_layer_tuples)

  # Now set up the HuBERT model.
  model = hubert.HuBERTModel(
      num_classes=num_classes,
      quantizer=quantizer_list,
      frontend=frontend,
      early_feature_extractor=early_fs,
      **model_config)
  variables = model.init(
      model_init_key,
      jnp.zeros((1, input_size)),
      train=False,
      mask_key=mask_key)
  model_state, params = variables.pop("params")

  # NOTE: https://github.com/deepmind/optax/issues/160
  params = params.unfreeze()

  # Define the learning rate schedule for HuBERT.
  # peak_scaling factor is such that if we multiply the initial learning rate
  # with it, we get the intended peak learning rate.
  peak_scaling_factor = learning_rate / start_learning_rate
  learning_rate = optax.piecewise_interpolate_schedule(
      "linear",
      init_value=start_learning_rate,
      boundaries_and_scales={
          int(num_train_steps / 2): peak_scaling_factor,
          num_train_steps: start_learning_rate
      })

  # Initialize optimizer and handle constraints
  std_to_fwhm = jnp.sqrt(2 * jnp.log(2)) / jnp.pi
  if frontend is None:
    optimizer = optax.adam(learning_rate=learning_rate)
  else:
    optimizer = optax.chain(
        optax.adam(learning_rate=learning_rate),
        optax.masked(
            project(0.0, 1.0), mask_by_name("spcen_smoothing_coef", params)),
        optax.masked(project(0.0, jnp.pi), mask_by_name("gabor_mean", params)),
        optax.masked(
            project(4 * std_to_fwhm, model.frontend.kernel_size * std_to_fwhm),
            mask_by_name("gabor_std", params)))
  opt_state = optimizer.init(params)

  # Load checkpoint
  ckpt = checkpoint.MultihostCheckpoint(workdir)
  train_state = TrainState(
      step=0, params=params, opt_state=opt_state, model_state=model_state)

  did_reload = False
  while not did_reload:
    try:
      train_state = ckpt.restore_or_initialize(train_state)
      did_reload = True
      break
    except tf.errors.NotFoundError:
      logging.warning(
          "Reloading from %s failed. Taking a nap and will try again.", workdir)
      time.sleep(5)
    except:  # pylint: disable=bare-except
      logging.warning(
          "Reloading from %s failed for some unexpected reason. Taking a nap "
          "and will try again.", workdir)
      time.sleep(5)

  if reload_quantizer_from:
    ckpt_to_reload = checkpoint.MultihostCheckpoint(reload_quantizer_from)
    did_reload = False
    while not did_reload:
      try:
        reloaded_quantizer = ckpt_to_reload.restore(None)
        did_reload = True
        break
      except tf.errors.NotFoundError:
        logging.warning(
            "Reloading from %s failed. Taking a nap and will try again.",
            reload_quantizer_from)
        time.sleep(5)
    print("reloaded_quantizer codebook {}".format(
        reloaded_quantizer["params"]["quantizer"]))
    train_state.params["quantizer"] = reloaded_quantizer["params"]["quantizer"]

  return ModelBundle(model, optimizer, key, ckpt), train_state


def train(model_bundle,
          train_state,
          train_dataset,
          num_train_steps: int,
          logdir: str,
          log_every_steps: int,
          checkpoint_every_steps: int,
          num_quantizer_pretrain_steps: int,
          quant_loss_mult: float,
          readout_loss_mult: float,
          reload_quantizer=False) -> None:
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
    quant_loss_mult: The multiplier for the quantizer loss in the combined loss
      used for training.
    readout_loss_mult: The multiplier for the readout loss in the combined loss
      used for training.
    reload_quantizer: Whether to reload a pre-trained quantizer. If this is the
      case, it is kept frozen.
  """
  if reload_quantizer and num_quantizer_pretrain_steps:
    raise ValueError("Cannot have both num_quantizer_steps being nonzero and "
                     "reload_quantizer being True.")

  train_iterator = train_dataset.as_numpy_iterator()
  train_metrics_collection = make_metrics_collection(
      "train___", model_bundle.model.alpha, quant_loss_mult, readout_loss_mult,
      model_bundle.model.readout_points, model_bundle.model.quantizer_points)

  def get_update_step(loss_key="train___loss"):

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
            taxonomy_loss_weight=model_bundle.model.taxonomy_loss_weight
        ).compute()
        loss = train_metrics[loss_key]
        return loss, (train_metrics, model_state)

      # model_state has only the batch_norm stats which only appear in the
      # late feature extractor (conformer).
      (_, (train_metrics, model_state)), grads = jax.value_and_grad(
          step, has_aux=True)(train_state.params, train_state.model_state)
      grads = jax.lax.pmean(grads, axis_name="batch")
      updates, opt_state = model_bundle.optimizer.update(
          grads, train_state.opt_state, train_state.params)

      params_after_update = optax.apply_updates(train_state.params, updates)

      train_state = TrainState(
          step=train_state.step + 1,
          params=params_after_update,
          opt_state=opt_state,
          model_state=model_state)
      return train_metrics, train_state

    return update_step

  if num_quantizer_pretrain_steps:
    quantizer_step = get_update_step("train___quantizer_loss")
  joint_step = get_update_step("train___loss")

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

      mask_key = random.split(mask_key, num=jax.local_device_count())
      step_key = random.split(step_key, num=jax.local_device_count())

      if step < num_quantizer_pretrain_steps:
        # Train only the quantizer.
        train_metrics, train_state = quantizer_step(step_key, batch,
                                                    train_state, mask_key)
      else:
        # Joint training.
        train_metrics, train_state = joint_step(step_key, batch, train_state,
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
  quant_loss_mult, readout_loss_mult = 1, 1
  valid_metrics = make_metrics_collection("valid___", model_bundle.model.alpha,
                                          quant_loss_mult, readout_loss_mult,
                                          model_bundle.model.readout_points,
                                          model_bundle.model.quantizer_points)

  @functools.partial(jax.pmap, axis_name="batch")
  def update_metrics(valid_metrics, batch, train_state):
    variables = {"params": train_state.params, **train_state.model_state}
    model_outputs = model_bundle.model.apply(
        variables, batch["audio"], train=False, mask_key=None)
    return model_outputs, valid_metrics.merge(
        valid_metrics.gather_from_model_output(
            outputs=model_outputs,
            label=batch["label"],
            genus=batch["genus"],
            family=batch["family"],
            order=batch["order"],
            taxonomy_loss_weight=model_bundle.model.taxonomy_loss_weight,
            axis_name="batch"))

  step = int(flax_utils.unreplicate(train_state.step))
  label_names = ("label", "genus", "family", "order")
  cmap_metrics = make_cmap_metrics_dict(label_names,
                                        model_bundle.model.readout_points)
  with reporter.timed("eval"):
    valid_metrics = flax_utils.replicate(valid_metrics.empty())
    for s, batch in enumerate(valid_dataset.as_numpy_iterator()):
      batch = jax.tree_map(np.asarray, batch)
      model_outputs, valid_metrics = update_metrics(valid_metrics, batch,
                                                    train_state)
      cmap_metrics = update_cmap_metrics_dict(label_names, cmap_metrics,
                                              model_outputs, batch,
                                              model_bundle.model.readout_points)
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
