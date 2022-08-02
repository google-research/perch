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

"""Adaptation and evaluation utilities for source-free domain adaptation."""

import abc
import functools
from typing import Dict, Optional, Tuple, Type

from chirp import train
from chirp.models import class_average
from clu import metric_writers
from clu import metrics as clu_metrics
from clu import periodic_actions
import flax
from flax import linen as nn
import flax.jax_utils as flax_utils
import jax
from jax import numpy as jnp
from ml_collections import config_dict
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds


@flax.struct.dataclass
class AdaptationState:
  """All useful states kept during adaptation.

  Unlike in train.py where TrainState contains a single model, adaptation
  methods may use several methods (e.g. a teacher and a student, or an
  auxiliary GAN). Therefore, model_params, model_states and opts_states are
  stored in Dict in which the keys refer to the name of the model. The key
  'main' will be used for evaluation.
  """
  step: int
  epoch: int
  models_params: Dict[str, flax.core.scope.VariableDict]
  model_states: Dict[str, flax.core.scope.FrozenVariableDict]
  opt_states: Optional[Dict[str, optax.OptState]] = None


@flax.struct.dataclass
class EvaluationState:
  step: int
  model_params: flax.core.scope.VariableDict
  model_state: flax.core.scope.FrozenVariableDict


@flax.struct.dataclass
class ModelBundle:
  model: nn.Module
  optimizer: Optional[optax.GradientTransformation]
  key: jnp.ndarray


class SFDAMethod(metaclass=abc.ABCMeta):
  """A template for any source-free domain adaptation method."""

  def initialize(
      self, source_dataset_info: tfds.core.DatasetInfo,
      model_config: config_dict.ConfigDict, rng_seed: int, input_size: int,
      pretrained_ckpt_dir: str, **method_kwargs
  ) -> Tuple[Dict[str, ModelBundle], AdaptationState, jax.random.PRNGKeyArray]:
    """Instantiate the models, states and key for adaptation and evaluation.

    Each source-free domain adaptation (SFDA) method may have a different way
    to instantiating the models. For instance, some models may have more than
    1 model, e.g. a teacher and a student, etc.. The default implementation
    provided here only instantiates a single `main` model, along with its
    optimizer. Methods requiring more elaborate initializations may override
    the method.

    Args:
      source_dataset_info: Info on the source dataset, used to retrieve the
        source model (e.g. number of classes).
      model_config: The model configuration used to build the model.
      rng_seed: The seed from which to instantiate the key.
      input_size: The temporal dimension of the input.
      pretrained_ckpt_dir: The checkpoint directory to the source model.
      **method_kwargs: Additional kwargs specific to each method.

    Returns:
      model_bundles: A dict of all model bundles
      adaptation_state: The AdaptationState used throught adaptation and
        evaluation.
      key: The jax random key used throughout the pipeline.
    """

    # Generate a random key
    key = jax.random.PRNGKey(rng_seed)

    # Load main classification model from pretrained checkpoint
    model_bundle, train_state = train.initialize_model(source_dataset_info,
                                                       model_config, rng_seed,
                                                       input_size, 0.,
                                                       pretrained_ckpt_dir)

    # Define the optimizer
    params = train_state.params
    std_to_fwhm = jnp.sqrt(2 * jnp.log(2)) / jnp.pi
    optimizer = optax.chain(
        optax.adam(learning_rate=method_kwargs["learning_rate"]),
        optax.masked(
            train.project(0.0, 1.0),
            train.mask_by_name("spcen_smoothing_coef", params)),
        optax.masked(
            train.project(0.0, jnp.pi),
            train.mask_by_name("gabor_mean", params)),
        optax.masked(
            train.project(0.0, jnp.pi),
            train.mask_by_name("gabor_mean", params)),
        optax.masked(
            train.project(
                4 * std_to_fwhm,
                model_bundle.model.frontend.kernel_size * std_to_fwhm),
            train.mask_by_name("gabor_std", params)))
    opt_state = optimizer.init(params)

    # Recycle a lot from the train_state to create the adaptation state.
    # We recreate a new key, and a new optimizer.
    model_bundles = {"main": ModelBundle(model_bundle.model, optimizer, key)}
    opt_states = {"main": opt_state}
    model_states = {"main": train_state.model_state}
    models_params = {"main": train_state.params}

    adaptation_state = AdaptationState(
        step=0,
        epoch=0,
        models_params=models_params,
        opt_states=opt_states,
        model_states=model_states)

    return model_bundles, adaptation_state, key

  def get_adaptation_metrics(self) -> Type[clu_metrics.Collection]:
    """Obtain metrics that will be monitored during adaptation.

    Kept as part of the method, as each method may be interested in monitoring
    different methods. Defaults to the common metrics.

    Returns:
      The metrics collection used during adaptation.
    """

    return get_common_metrics("adaptation___")

  @abc.abstractmethod
  def do_epoch(self, key, model_bundles, adaptation_state, adaptation_dataset,
               metrics_collection, writer, reporter) -> AdaptationState:
    """Perform the adaptation for one epoch."""

  pass


def perform_adaptation(key: jax.random.PRNGKeyArray, da_method: SFDAMethod,
                       adaptation_state: AdaptationState,
                       adaptation_dataset: tf.data.Dataset,
                       model_bundles: Dict[str, ModelBundle], logdir: str,
                       num_epochs: int):
  """Runs adaptation for some adaptation method and some dataset."""

  # Initialize data
  adaptation_metrics_collection = da_method.get_adaptation_metrics()

  # Logging
  writer = metric_writers.create_default_writer(logdir)
  reporter = periodic_actions.ReportProgress(writer=writer)
  for _ in range(num_epochs):
    adaptation_state = da_method.do_epoch(
        key=key,
        model_bundles=model_bundles,
        adaptation_state=adaptation_state,
        adaptation_dataset=adaptation_dataset,
        metrics_collection=adaptation_metrics_collection,
        writer=writer,
        reporter=reporter)
    writer.flush()
  writer.close()
  return adaptation_state


def get_common_metrics(prefix: str) -> Type[clu_metrics.Collection]:
  """Obtain a common set of metrics, including a label-based supervised loss."""
  metrics_dict = {}
  taxo_keys = ["label"]
  for key in taxo_keys:
    metrics_dict.update({
        key + "_xentropy":
            clu_metrics.Average.from_fun(
                functools.partial(train.keyed_cross_entropy, key=key)),
        key + "_map":
            clu_metrics.Average.from_fun(
                functools.partial(train.keyed_map, key=key)),
        key + "_cmap":
            class_average.ClassAverage.from_fun(
                functools.partial(train.keyed_cmap, key=key)),
    })

  # Define loss here
  metrics_dict["supervised_loss"] = metrics_dict["label_xentropy"]
  metrics_dict = {prefix + k: v for k, v in metrics_dict.items()}
  return clu_metrics.Collection.create(**metrics_dict)


def evaluate(
    model_bundles: Dict[str, ModelBundle],
    adaptation_state: AdaptationState,
    eval_dataset: tf.data.Dataset,
    logdir: str,
) -> None:
  """The evaluation loop.

  Args:
    model_bundles: The model bundles used by the method.
    adaptation_state: The adaptation_state after adapation has happened.
    eval_dataset: The dataset used for evaluation.
    logdir: Directory to log evaluation metrics.
  """
  # We keep the `main` model for evaluation
  evaluation_state = EvaluationState(
      step=0,
      model_params=adaptation_state.models_params["main"],
      model_state=adaptation_state.model_states["main"])
  model = model_bundles["main"].model

  # Define logging
  writer = metric_writers.create_default_writer(logdir)
  reporter = periodic_actions.ReportProgress(
      writer=writer, num_train_steps=len([_ for _ in iter(eval_dataset)]))

  # Define logging
  valid_metrics = get_common_metrics("validation___")

  # Replicate everything across devices
  evaluation_state = flax_utils.replicate(evaluation_state)
  valid_metrics = flax_utils.replicate(valid_metrics.empty())

  @functools.partial(jax.pmap, axis_name="batch")
  def update_metrics(metric_collection, batch,
                     evaluation_state) -> clu_metrics.Collection:

    variables = {
        "params": evaluation_state.model_params,
        **evaluation_state.model_state
    }
    model_outputs = model.apply(variables, batch["audio"], train=False)
    return metric_collection.merge(
        metric_collection.gather_from_model_output(
            outputs=model_outputs,
            label=batch["label"],
            genus=batch["genus"],
            family=batch["family"],
            order=batch["order"]))

  with reporter.timed("eval"):
    for step, batch in enumerate(eval_dataset.as_numpy_iterator()):
      batch = {
          k: v
          for k, v in batch.items()
          if k in ["audio", "label", "genus", "family", "order"]
      }
      batch = jax.tree_map(np.asarray, batch)
      valid_metrics = update_metrics(valid_metrics, batch, evaluation_state)
      reporter(step)
      writer.write_scalars(step,
                           flax_utils.unreplicate(valid_metrics).compute())
      writer.flush()
    writer.close()
