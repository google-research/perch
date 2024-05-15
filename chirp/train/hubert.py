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

"""Training loop."""

import enum
import functools
import os
import time
from typing import Any, Callable

from absl import logging
from chirp.data import utils as data_utils
from chirp.models import frontend as frontend_models
from chirp.models import hubert
from chirp.models import layers
from chirp.models import metrics
from chirp.models import output
from chirp.models import quantizers
from chirp.taxonomy import class_utils
from chirp.train import train_utils
from clu import checkpoint
from clu import metric_writers
from clu import metrics as clu_metrics
from clu import periodic_actions
import flax
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


def filtered_hubert_loss_from_outputs(
    outputs: hubert.HubertOutput, keep_inds: jnp.ndarray, **unused_kwargs
) -> jnp.ndarray:
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


def hubert_loss_from_outputs(
    outputs: hubert.HubertOutput,
    alpha: float,
    hubert_loss_mult: float,
    **unused_kwargs,
) -> jnp.ndarray:
  """Cross entropy computed from model outputs."""
  mask_idc = outputs.mask_idc
  # Compute the loss on the unmasked and masked frames separately.
  loss_u = filtered_hubert_loss_from_outputs(
      outputs, jnp.where(mask_idc, False, True)
  )
  loss_m = filtered_hubert_loss_from_outputs(
      outputs, jnp.where(mask_idc, True, False)
  )
  return hubert_loss_mult * (alpha * loss_m + (1 - alpha) * loss_u)


def quantizer_loss(
    outputs: hubert.HubertOutput, quant_loss_mult: float, **unused_kwargs
) -> jnp.ndarray:
  """Get quantization loss from model outputs."""
  del unused_kwargs
  # [bsz, sz, csz] or [bsz, sz, 1] (depending on the quantizer).
  quant_loss = outputs.quantization_loss
  quant_loss = jnp.squeeze(jnp.mean(quant_loss, -1))  # pytype: disable=wrong-arg-types  # jnp-type
  # [bsz, sz].
  return quant_loss * quant_loss_mult


def taxonomy_cross_entropy(
    outputs: hubert.HubertOutput,
    taxonomy_loss_weight: float,
    label: jnp.ndarray,
    genus: jnp.ndarray | None = None,
    family: jnp.ndarray | None = None,
    order: jnp.ndarray | None = None,
    **unused_kwargs,
) -> jnp.ndarray:
  """Computes mean cross entropy across taxonomic labels."""

  def aggregate_losses(preds, target):
    # Iterate over the label made from different readout points.
    losses = []
    for l in preds:
      losses.append(
          jnp.mean(optax.sigmoid_binary_cross_entropy(l, target), axis=-1)
      )
    return jnp.sum(jnp.stack(losses, axis=0), axis=0)

  mean = aggregate_losses(outputs.label, label)

  if taxonomy_loss_weight != 0:
    mean += taxonomy_loss_weight * aggregate_losses(outputs.genus, genus)
    mean += taxonomy_loss_weight * aggregate_losses(outputs.family, family)
    mean += taxonomy_loss_weight * aggregate_losses(outputs.order, order)
  return mean


def supervised_loss(
    outputs: hubert.HubertOutput,
    taxonomy_loss_weight: float,
    readout_loss_mult: float,
    label: jnp.ndarray,
    genus: jnp.ndarray | None = None,
    family: jnp.ndarray | None = None,
    order: jnp.ndarray | None = None,
    **unused_kwargs,
) -> jnp.ndarray:
  """Compute classification loss for all taxonomy heads."""
  del unused_kwargs
  if not readout_loss_mult:
    # Avoid computing the loss if not needed.
    # [bsz, sz].
    return jnp.zeros(outputs.logits[0].shape[:-1])
  loss = taxonomy_cross_entropy(
      outputs, taxonomy_loss_weight, label, genus, family, order
  )  # [bsz].
  # Make it [bsz, sz] so that it can be element-wise added to other losses.
  sz = outputs.logits[0].shape[-2]
  loss = jnp.repeat(jnp.expand_dims(loss, axis=-1), axis=-1, repeats=sz)
  return loss * readout_loss_mult


def keyed_cross_entropy(
    key: str,
    outputs: hubert.HubertOutput,
    readout_index: int = 0,
    **kwargs,
) -> jnp.ndarray | None:
  """Cross entropy for the specified taxonomic label set."""
  outputs = getattr(outputs, key)
  outputs = outputs[readout_index]

  ce = optax.sigmoid_binary_cross_entropy(outputs, kwargs[key])
  return ce


def keyed_map(
    key: str, outputs: hubert.HubertOutput, readout_index: int = 0, **kwargs
) -> jnp.ndarray | None:
  outputs = getattr(outputs, key)
  outputs = outputs[readout_index]
  return metrics.average_precision(scores=outputs, labels=kwargs[key])


def final_loss(
    outputs: hubert.HubertOutput,
    alpha: float,
    quant_loss_mult: float,
    readout_loss_mult: float,
    hubert_loss_mult: float,
    **kwargs_for_supervised,
) -> jnp.ndarray | None:
  """Get the final loss to use for training."""
  # [bsz, sz].
  quant_loss = quantizer_loss(outputs, quant_loss_mult)
  if not hubert_loss_mult and not readout_loss_mult:
    return quant_loss

  # [bsz, sz].
  readout_loss = supervised_loss(
      outputs, readout_loss_mult=readout_loss_mult, **kwargs_for_supervised
  )

  # [nq, ns, bsz, sz].
  hubert_loss = hubert_loss_from_outputs(
      outputs, alpha, hubert_loss_mult=hubert_loss_mult
  )

  # Make the shapes match so that these losses can be added elementwise.
  nq, ns, _, _ = hubert_loss.shape
  quant_loss = jnp.repeat(jnp.expand_dims(quant_loss, 0), ns, axis=0)
  quant_loss = jnp.repeat(jnp.expand_dims(quant_loss, 0), nq, axis=0)
  readout_loss = jnp.repeat(jnp.expand_dims(readout_loss, 0), ns, axis=0)
  readout_loss = jnp.repeat(jnp.expand_dims(readout_loss, 0), nq, axis=0)

  return quant_loss + hubert_loss + readout_loss


def cluster_targets_metrics(
    outputs: hubert.HubertOutput, key: str, **unused_kwargs
) -> jnp.ndarray | None:
  """Get the final loss to use for training."""
  del unused_kwargs
  assert key.startswith((
      "n_masked_per_sample",
      "n_per_cluster",
      "max_per_cluster",
      "min_per_cluster",
      "h_diversity",
  ))
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
        "h_diversity_{}".format(i): h_diversity,
    })
  return ret[key]


def get_train_metrics(
    keys: list[str],
    num_labels: dict[str, int],
    alpha: float,
    readout_loss_mult: float,
    hubert_loss_mult: float,
    quantizer_points: list[int],
    readout_points: list[int],
) -> dict[str, type[clu_metrics.Metric]]:
  """Create a collection of metrics with cross-entropy and average precision."""

  metrics_ = {
      "loss": clu_metrics.Average.from_output("loss"),
      "learning_rate": clu_metrics.LastValue.from_output("learning_rate"),
      "hubert_loss": clu_metrics.Average.from_fun(
          functools.partial(
              hubert_loss_from_outputs,
              alpha=alpha,
              hubert_loss_mult=hubert_loss_mult,
          )
      ),
      "quantizer_loss": clu_metrics.Average.from_output("quantizer_loss"),
      "supervised_loss": clu_metrics.Average.from_fun(
          functools.partial(
              supervised_loss, readout_loss_mult=readout_loss_mult
          )
      ),
  }

  for i, block_ind in enumerate(quantizer_points):
    block_name = "late_fs_{}".format(block_ind) if block_ind >= 0 else "earlyfs"
    metrics_.update({
        "n_per_cluster_{}".format(block_name): clu_metrics.Average.from_fun(
            functools.partial(
                cluster_targets_metrics, key="n_per_cluster_{}".format(i)
            )
        ),
        "max_per_cluster_{}".format(block_name): clu_metrics.Average.from_fun(
            functools.partial(
                cluster_targets_metrics, key="max_per_cluster_{}".format(i)
            )
        ),
        "min_per_cluster_{}".format(block_name): clu_metrics.Average.from_fun(
            functools.partial(
                cluster_targets_metrics, key="min_per_cluster_{}".format(i)
            )
        ),
        "h_diversity_{}".format(block_name): clu_metrics.Average.from_fun(
            functools.partial(
                cluster_targets_metrics, key="h_diversity_{}".format(i)
            )
        ),
    })

  for i, block_ind in enumerate(readout_points):
    for key in keys:
      metrics_.update({
          f"{key}_{block_ind}_xentropy": train_utils.MultiAverage.create(
              num_labels[key]
          ).from_fun(
              functools.partial(keyed_cross_entropy, key=key, readout_index=i)
          ),
          f"{key}_{block_ind}_map": clu_metrics.Average.from_fun(
              functools.partial(keyed_map, key=key, readout_index=i)
          ),
      })

  return metrics_


class LearningRateSchedule(enum.Enum):
  """A point in the architecture to add a quantizer."""

  PIECEWISE_LINEAR = "piecewise_linear"
  PIECEWISE_COSINE = "piecewise_cosine"
  COSINE_DECAY = "cosine_decay"


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
        lambda p, u: jnp.clip(p + u, min_value, max_value) - p, params, updates
    )

  return optax.stateless(clip_value)


def initialize_model(
    model_config: config_dict.ConfigDict,
    rng_seed: int,
    input_shape: tuple[int, ...],
    learning_rate: float,
    start_learning_rate: float,
    workdir: str,
    learning_rate_schedule: LearningRateSchedule,
    num_train_steps: int,
    quantizer_config: config_dict.ConfigDict,
    base_quantizer_config: config_dict.ConfigDict,
    frontend_config: config_dict.ConfigDict,
    early_fs_config: config_dict.ConfigDict,
    reload_quantizer_from: str,
    reload_hubert_from: str,
    reload_hubert_omit_quantizers: bool,
    target_class_list: str,
    early_fs_class: Callable[..., Any] | None = layers.EarlyFeatureExtractor,
    **unused_kwargs,
):
  """Creates model for training, eval, or inference."""
  del unused_kwargs
  # Initialize random number generator
  key = random.PRNGKey(rng_seed)

  # Load model
  model_init_key, mask_key = random.split(key)
  class_lists = class_utils.get_class_lists(target_class_list, True)
  num_classes = {k: len(v.classes) for (k, v) in class_lists.items()}

  # Initialize the quantizer.
  if quantizer_config.use_entropy_quantizer:
    kwargs = {
        "num_centroids": base_quantizer_config.num_centroids,
        "gamma": base_quantizer_config.gamma,
    }
    quantizer_class = quantizers.VectorQuantizerEnt
  else:
    kwargs = {
        "num_centroids": base_quantizer_config.num_centroids,
        "demean": True,
        "rescale": True,
    }
    quantizer_class = quantizers.VectorQuantizer
  quantizer_list = []
  for _ in range(len(model_config.quantizer_points)):
    quantizer = None
    if (
        quantizer_config.strategy
        == quantizers.QuantizationStrategy.PRODUCT_QUANTIZATION.value
    ):
      base_quantizers = [
          quantizer_class(**kwargs)
          for _ in range(quantizer_config.num_sections)
      ]
      quantizer = quantizers.ProductQuantizer(base_quantizers=base_quantizers)
    elif (
        quantizer_config.strategy
        == quantizers.QuantizationStrategy.RESIDUAL_QUANTIZATION.value
    ):
      base_quantizers = [
          quantizer_class(**kwargs)
          for _ in range(quantizer_config.num_sections)
      ]
      quantizer = quantizers.ResidualQuantizer(quantizers=base_quantizers)
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
        scaling_config=frontend_config.scaling_config,
    )

  # Initialize the early feature extractor.
  if model_config.use_raw_audio:
    if early_fs_config.omit_earlyfs:
      raise ValueError(
          "Expected the early feature extractor to be provided if "
          "using raw audio."
      )
    if (
        hubert.QuantizerPoints.FRONTEND.value in model_config.quantizer_points
        and frontend is None
    ):
      raise ValueError(
          "Expected frontend to be provided in order to "
          "perform quantization on the frontend outputs."
      )

    # The original architecture, from wav2vec, which leads to 500 frames.
    conv_layer_tuples = tuple([
        (512, 10, 5),
        (512, 3, 2),
        (512, 3, 2),
        (512, 3, 2),
        (512, 3, 2),
        (512, 2, 2),
        (512, 2, 2),
    ])
    early_fs = early_fs_class(
        dropout_prob=early_fs_config.dropout_prob,
        activation=early_fs_config.activation,
        conv_layer_tuples=conv_layer_tuples,
        deprecated_group_conv=early_fs_config.deprecated_group_conv,
    )

  else:
    if early_fs_config.omit_earlyfs:
      early_fs = None
    else:
      if early_fs_config.num_frames not in [125, 63, 32, 16]:
        raise ValueError(
            "Expected early_fs_config.num_frames to be 125, 63, 32 or 16."
        )
      conv_layer_tuples = None
      if frontend is None:
        # Their original architecture led to 500 frames which caused OOM.
        # Added 2 additional conv layers with stride 2 which makes it 125.
        # Still was getting OOM with this with batch size 128, so reduced to 64.
        conv_layer_tuples = tuple([
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
            (512, 2, 2),
            (512, 2, 2),
        ])
      else:
        nf = 512
        if early_fs_config.num_frames == 125:
          # With this configuration, the number of frames is reduced from 500 to
          # 125 and the framerate is reduced from 100Hz (which the frontend
          # outputs) 25Hz.
          conv_layer_tuples = tuple([
              (nf, 10, 2),
              (nf, 3, 2),
              (nf, 3, 1),
              (nf, 3, 1),
              (nf, 3, 1),
              (nf, 2, 1),
              (nf, 2, 1),
          ])
        elif early_fs_config.num_frames == 63:
          conv_layer_tuples = tuple([
              (nf, 10, 2),
              (nf, 3, 2),
              (nf, 3, 2),
              (nf, 3, 1),
              (nf, 3, 1),
              (nf, 2, 1),
              (nf, 2, 1),
          ])
        elif early_fs_config.num_frames == 32:
          conv_layer_tuples = tuple([
              (nf, 10, 2),
              (nf, 3, 2),
              (nf, 3, 2),
              (nf, 3, 2),
              (nf, 3, 1),
              (nf, 2, 1),
              (nf, 2, 1),
          ])
        elif early_fs_config.num_frames == 16:
          conv_layer_tuples = tuple([
              (nf, 10, 2),
              (nf, 3, 2),
              (nf, 3, 2),
              (nf, 3, 2),
              (nf, 3, 2),
              (nf, 2, 1),
              (nf, 2, 1),
          ])
      early_fs = early_fs_class(
          dropout_prob=early_fs_config.dropout_prob,
          activation=early_fs_config.activation,
          conv_layer_tuples=conv_layer_tuples,
      )

  # Now set up the HuBERT model.
  model = hubert.HuBERTModel(
      num_classes=num_classes,
      quantizer=quantizer_list,
      frontend=frontend,
      early_feature_extractor=early_fs,
      **model_config,
  )
  variables = model.init(
      model_init_key,
      jnp.zeros((1,) + input_shape),
      train=False,
      mask_key=mask_key,
      train_mode_quantizer=False,
  )
  model_state, params = flax.core.pop(variables, "params")

  # NOTE: https://github.com/deepmind/optax/issues/160
  params = flax.core.unfreeze(params)

  # Define the learning rate schedule for HuBERT.
  learning_rate_schedule = LearningRateSchedule(learning_rate_schedule)
  if learning_rate_schedule is LearningRateSchedule.PIECEWISE_LINEAR:
    # peak_scaling factor is such that if we multiply the initial learning rate
    # with it, we get the intended peak learning rate.
    peak_scaling_factor = learning_rate / start_learning_rate
    learning_rate = optax.piecewise_interpolate_schedule(
        "linear",
        init_value=start_learning_rate,
        boundaries_and_scales={
            int(0.08 * num_train_steps): peak_scaling_factor,
            num_train_steps: start_learning_rate,
        },
    )
  elif learning_rate_schedule is LearningRateSchedule.COSINE_DECAY:
    # only `start_learning_rate` and `num_train_steps` are used in this case.
    learning_rate = optax.cosine_decay_schedule(
        init_value=start_learning_rate,
        decay_steps=num_train_steps,
    )
  else:
    raise ValueError("unknown learning rate schedule")

  # Initialize optimizer and handle constraints
  optimizer = optax.adam(learning_rate=learning_rate)
  opt_state = optimizer.init(params)

  # Load checkpoint
  ckpt = checkpoint.MultihostCheckpoint(workdir)
  train_state = train_utils.TrainState(
      step=0, params=params, opt_state=opt_state, model_state=model_state
  )

  did_reload = False
  num_attempts = 0
  while not did_reload and num_attempts < 5:
    try:
      train_state = ckpt.restore_or_initialize(train_state)
      did_reload = True
      break
    except tf.errors.NotFoundError:
      logging.warning(
          "Reloading from %s failed. Taking a nap and will try again.", workdir
      )
      time.sleep(5)
    except:  # pylint: disable=bare-except
      logging.warning(
          (
              "Reloading from %s failed for some unexpected reason. Taking a"
              " nap and will try again."
          ),
          workdir,
      )
      time.sleep(5)
    num_attempts += 1

  if reload_quantizer_from:
    ckpt_to_reload = checkpoint.MultihostCheckpoint(reload_quantizer_from)
    did_reload = False
    num_attempts = 0
    reloaded_quantizer = None
    while not did_reload and num_attempts < 5:
      try:
        reloaded_quantizer = ckpt_to_reload.restore(None)
        did_reload = True
        break
      except tf.errors.NotFoundError:
        logging.warning(
            "Reloading from %s failed. Taking a nap and will try again.",
            reload_quantizer_from,
        )
        time.sleep(5)
      num_attempts += 1
    if reloaded_quantizer is None:
      raise RuntimeError(
          "Unable to reload quantizer from %s." % reload_quantizer_from
      )
    if "quantizer" in reloaded_quantizer["params"].keys():
      quantizer_key = "quantizer"
    elif "quantizer_0" in reloaded_quantizer["params"].keys():
      quantizer_key = "quantizer_0"
    else:
      raise RuntimeError(
          "Unsure which parameters correspond to the quantizer, "
          "so unable to reload it. The reloaded params do not contain a key "
          "'quantizer' nor 'quantizer_0'."
      )
    train_state.params[quantizer_key] = reloaded_quantizer["params"][  # pytype: disable=unsupported-operands  # py310-upgrade
        quantizer_key
    ]

  if reload_hubert_from:
    ckpt_to_reload = checkpoint.MultihostCheckpoint(reload_hubert_from)
    did_reload = False
    num_attempts = 0
    reloaded_hubert = None
    while not did_reload and num_attempts < 5:
      try:
        reloaded_hubert = ckpt_to_reload.restore(None)
        did_reload = True
        break
      except tf.errors.NotFoundError:
        logging.warning(
            "Reloading from %s failed. Taking a nap and will try again.",
            reload_hubert_from,
        )
        time.sleep(5)
      num_attempts += 1
    if reloaded_hubert is None:
      raise RuntimeError(
          "Unable to reload HuBERT from %s." % reload_hubert_from
      )
    logging.info(
        "Reloaded HuBERT params with keys %s", reloaded_hubert["params"].keys()
    )
    for k, v in reloaded_hubert["params"].items():
      # Since this reloading is done for continuing to train HuBERT with a new
      # quantizer (in a different space), we assume it's best to re-initialize
      # the projections between the features and these new codes.
      if reload_hubert_omit_quantizers and (
          k.startswith("codes_proj")
          or k.startswith("final_proj")
          or k.startswith("quantizer")
      ):
        logging.info("Ignoring HuBERT parameters for key %s.", k)
        continue
      train_state.params[k] = (
          v  # pytype: disable=unsupported-operands  # py310-upgrade
      )
      logging.info("Assigned reloaded HuBERT parameters for key %s.", k)

  return (
      train_utils.ModelBundle(
          model=model, key=key, ckpt=ckpt, optimizer=optimizer
      ),
      train_state,
      learning_rate,
  )


def train(
    model_bundle,
    train_state,
    learning_rate_schedule,
    train_dataset,
    num_train_steps: int,
    logdir: str,
    log_every_steps: int,
    checkpoint_every_steps: int,
    num_quantizer_pretrain_steps: int,
    quant_loss_mult: float,
    readout_loss_mult: float,
    hubert_loss_mult: float,
    reload_quantizer=False,
) -> None:
  """Train a model.

  Args:
    model_bundle: Static objects for conducting the experiment.
    train_state: Initial train_utils.TrainState.
    learning_rate_schedule: The schedule for the learning rate.
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
    hubert_loss_mult: The multiplier for the HuBERT loss in the combined loss
      used for training.
    reload_quantizer: Whether to reload a pre-trained quantizer. If this is the
      case, it is kept frozen.
  """
  if reload_quantizer and num_quantizer_pretrain_steps:
    raise ValueError(
        "Cannot have both num_quantizer_steps being nonzero and "
        "reload_quantizer being True."
    )

  if train_dataset is None:
    raise ValueError("train_dataset is None.")
  train_iterator = train_dataset.as_numpy_iterator()
  taxonomy_keys = ["label"]
  taxonomy_loss_weight = model_bundle.model.taxonomy_loss_weight
  if taxonomy_loss_weight != 0.0:
    taxonomy_keys += train_utils.TAXONOMY_KEYS
  train_metrics_collection = train_utils.NestedCollection.create(
      **get_train_metrics(
          taxonomy_keys,
          model_bundle.model.num_classes,
          alpha=model_bundle.model.alpha,
          readout_loss_mult=readout_loss_mult,
          hubert_loss_mult=hubert_loss_mult,
          quantizer_points=model_bundle.model.quantizer_points,
          readout_points=model_bundle.model.readout_points,
      )
  )

  @functools.partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=0)
  def update_step(quantizer_pretrain, key, batch, train_state, mask_key):
    dropout_key, low_pass_key = random.split(key)

    def step(params, model_state):
      variables = {"params": params, **model_state}
      x = jnp.squeeze(batch["audio"])
      model_outputs, model_state = model_bundle.model.apply(
          variables,
          x,
          train=True,
          mask_key=mask_key,
          train_mode_quantizer=True,
          mutable=list(model_state.keys()),
          rngs={
              "dropout": dropout_key,
              "low_pass": low_pass_key,
          },
      )
      quantizer_loss_ = quantizer_loss(
          model_outputs, quant_loss_mult=quant_loss_mult
      )
      final_loss_ = final_loss(
          model_outputs,
          taxonomy_loss_weight=taxonomy_loss_weight,
          alpha=model_bundle.model.alpha,
          quant_loss_mult=quant_loss_mult,
          readout_loss_mult=readout_loss_mult,
          hubert_loss_mult=hubert_loss_mult,
          **batch,
      )
      train_metrics = train_metrics_collection.gather_from_model_output(
          outputs=model_outputs,
          loss=final_loss_,
          quantizer_loss=quantizer_loss_,
          learning_rate=learning_rate_schedule(train_state.step),
          taxonomy_loss_weight=taxonomy_loss_weight,
          **batch,
          # CmAP expects logits to be passed as dict instead of dataclass
          **output.logits(model_outputs),
      )
      loss = quantizer_loss_ if quantizer_pretrain else final_loss_
      return jnp.mean(loss), (train_metrics, model_state)

    # model_state has only the batch_norm stats which only appear in the
    # late feature extractor (conformer).
    grads, (train_metrics, model_state) = jax.grad(step, has_aux=True)(
        train_state.params, train_state.model_state
    )
    grads = jax.lax.pmean(grads, axis_name="batch")
    updates, opt_state = model_bundle.optimizer.update(
        grads, train_state.opt_state, train_state.params
    )

    params = optax.apply_updates(train_state.params, updates)

    train_state = train_utils.TrainState(
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

      step_key, mask_key, key = random.split(key, num=3)

      mask_key = random.split(mask_key, num=jax.local_device_count())
      step_key = random.split(step_key, num=jax.local_device_count())

      quantizer_pretrain = step < num_quantizer_pretrain_steps
      train_metrics, train_state = update_step(
          quantizer_pretrain, step_key, batch, train_state, mask_key
      )

      if step % log_every_steps == 0:
        train_utils.write_metrics(
            writer,
            step,
            flax_utils.unreplicate(train_metrics).compute(prefix="train"),
        )
      reporter(step)

    if (step + 1) % checkpoint_every_steps == 0 or step == num_train_steps:
      with reporter.timed("checkpoint"):
        model_bundle.ckpt.save(flax_utils.unreplicate(train_state))
  writer.close()


def evaluate(
    model_bundle: train_utils.ModelBundle,
    train_state: train_utils.TrainState,
    learning_rate_schedule: optax.Schedule,
    valid_dataset: tf.data.Dataset,
    workdir: str,
    num_train_steps: int,
    eval_steps_per_checkpoint: int | None = None,
    train_mode_at_eval: bool | None = False,
    mask_at_eval: bool | None = False,
    name: str = "valid",
    eval_sleep_s: int = EVAL_LOOP_SLEEP_S,
):
  """Run evaluation."""
  quant_loss_mult, readout_loss_mult, hubert_loss_mult = 1, 1, 1
  taxonomy_keys = ["label"]
  taxonomy_loss_weight = model_bundle.model.taxonomy_loss_weight
  if taxonomy_loss_weight != 0.0:
    taxonomy_keys += train_utils.TAXONOMY_KEYS
  metrics_ = get_train_metrics(
      taxonomy_keys,
      model_bundle.model.num_classes,
      alpha=model_bundle.model.alpha,
      readout_loss_mult=readout_loss_mult,
      hubert_loss_mult=hubert_loss_mult,
      quantizer_points=model_bundle.model.quantizer_points,
      readout_points=model_bundle.model.readout_points,
  )
  rank_metrics = {}
  for key in taxonomy_keys:
    rank_metrics[f"{key}_cmap"] = (
        (f"{key}_logits", key),
        metrics.cmap,
    )
    rank_metrics[f"{key}_roc_auc"] = (
        (f"{key}_logits", key),
        metrics.roc_auc,
    )
  metrics_["rank_metrics"] = train_utils.CollectingMetrics.from_funs(
      **rank_metrics
  )
  valid_metrics_collection = train_utils.NestedCollection.create(**metrics_)

  @functools.partial(jax.pmap, axis_name="batch")
  def get_metrics(batch, train_state, mask_key):
    variables = {"params": train_state.params, **train_state.model_state}
    mutable = (
        list(train_state.model_state.keys()) if train_mode_at_eval else False
    )
    model_outputs = model_bundle.model.apply(
        variables,
        batch["audio"],
        train=train_mode_at_eval,
        mask_key=mask_key,
        train_mode_quantizer=False,
        mutable=mutable,
    )
    if mutable:
      # Both model outputs and state are returned if `mutable` was given.
      model_outputs = model_outputs[0]
    loss = final_loss(
        model_outputs,
        taxonomy_loss_weight=taxonomy_loss_weight,
        alpha=model_bundle.model.alpha,
        quant_loss_mult=quant_loss_mult,
        readout_loss_mult=readout_loss_mult,
        hubert_loss_mult=hubert_loss_mult,
        **batch,
    )
    return valid_metrics_collection.gather_from_model_output(
        outputs=model_outputs,
        loss=loss,
        quantizer_loss=quantizer_loss(model_outputs, quant_loss_mult),
        learning_rate=learning_rate_schedule(train_state.step),
        taxonomy_loss_weight=taxonomy_loss_weight,
        # TODO(bartvm): This only calculates CmAP over the first readout layer
        label_logits=model_outputs.label[0],
        **batch,
    )

  writer = metric_writers.create_default_writer(workdir)
  reporter = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer
  )
  for train_state in train_utils.checkpoint_iterator(
      train_state, model_bundle.ckpt, workdir, num_train_steps, eval_sleep_s
  ):
    step = int(train_state.step)
    key = model_bundle.key
    with reporter.timed("eval"):
      valid_metrics = valid_metrics_collection.empty()
      for s, batch in enumerate(valid_dataset.as_numpy_iterator()):
        batch = jax.tree.map(np.asarray, batch)
        mask_key = None
        if mask_at_eval:
          mask_key, key = random.split(key)
          mask_key = random.split(mask_key, num=jax.local_device_count())
        new_valid_metrics = get_metrics(
            batch, flax_utils.replicate(train_state), mask_key
        )
        valid_metrics = valid_metrics.merge(
            flax_utils.unreplicate(new_valid_metrics)
        )
        if (
            eval_steps_per_checkpoint is not None
            and s >= eval_steps_per_checkpoint
        ):
          break

      # Log validation loss
      train_utils.write_metrics(
          writer, step, valid_metrics.compute(prefix=name)
      )
    writer.flush()


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

  train_dataset, valid_dataset, dataset_info = None, None, None
  if mode == "train":
    train_dataset, dataset_info = data_utils.get_dataset(
        is_train=True,
        tf_data_service_address=tf_data_service_address,
        **config.train_dataset_config,
    )
  elif mode in ["eval", "tune_eval_hypers"]:
    valid_dataset, dataset_info = data_utils.get_dataset(
        **config.eval_dataset_config
    )
  elif mode == "export":
    valid_dataset, dataset_info = None, None

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

  reload_quantizer = False
  if config.init_config.reload_quantizer_from:
    reload_quantizer = True

  # Adjust the multiplier of the quantizer loss such that the quantizer gets the
  # intended starting learning rate.
  quant_start_lr = config.init_config.quant_start_learning_rate
  start_lr = config.init_config.start_learning_rate
  quant_loss_mult = quant_start_lr / start_lr
  quant_loss_mult *= config.train_config.quant_loss_mult

  # Initialize.
  if mode == "tune_eval_hypers":
    # Here, workdir is provided in the init config.
    model_bundle, train_state, learning_rate_schedule = initialize_model(
        num_train_steps=config.train_config.num_train_steps,
        **config.init_config,
    )
  else:
    model_bundle, train_state, learning_rate_schedule = initialize_model(
        workdir=workdir,
        num_train_steps=config.train_config.num_train_steps,
        **config.init_config,
    )

  if mode == "train":
    train(
        model_bundle,
        train_state,
        learning_rate_schedule,
        train_dataset,
        reload_quantizer=reload_quantizer,
        logdir=workdir,
        num_train_steps=config.train_config.num_train_steps,
        log_every_steps=config.train_config.log_every_steps,
        checkpoint_every_steps=config.train_config.checkpoint_every_steps,
        num_quantizer_pretrain_steps=config.train_config.num_quantizer_pretrain_steps,
        quant_loss_mult=quant_loss_mult,
        readout_loss_mult=config.train_config.readout_loss_mult,
        hubert_loss_mult=config.train_config.hubert_loss_mult,
    )

  elif mode == "tune_eval_hypers":
    # Running a single round of evaluation (as opposed to running eval in a
    # loop whenever a new checkpoint is produced).
    # This is used to tune HuBERT's evaluation hypers once.
    train_state = model_bundle.ckpt.restore(train_state)
    evaluate(
        model_bundle,
        flax_utils.replicate(train_state),
        learning_rate_schedule,
        valid_dataset,
        workdir=workdir,
        train_mode_at_eval=config.eval_config.train_mode_at_eval,
        mask_at_eval=config.eval_config.mask_at_eval,
        name=name,
        # Setting num_train_steps=0 will run eval exactly once.
        num_train_steps=0,
    )

  elif mode == "eval":
    evaluate(
        model_bundle,
        train_state,
        learning_rate_schedule,
        valid_dataset,
        workdir=workdir,
        name=name,
        **config.eval_config,
    )

  elif mode == "export":
    raise NotImplementedError("Export mode is not implemented for Hubert.")
